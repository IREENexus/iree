# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_bitcode_library()
#
# Builds an LLVM bitcode library from an input file via clang
#
# Parameters:
# NAME: Name of target (see Note).
# SRCS: Source files. Headers go here as well, as in iree_cc_library. There is
#       no concept of public headers (HDRS) here.
# COPTS: additional flags to pass to clang.
# OUT: Output file name (defaults to NAME.bc).
function(iree_bitcode_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT;ARCH"
    "INTERNAL_HDRS;SRCS;COPTS"
    ${ARGN}
  )

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.bc")
  endif()

  # Produce an empty file if the compiler wouldn't use bitcode for this arch anyway.
  iree_compiler_targeting_iree_arch(_IREE_COMPILER_TARGETING_THIS_ARCH "${_RULE_ARCH}")
  if (NOT _IREE_COMPILER_TARGETING_THIS_ARCH)
    iree_make_empty_file("${_OUT}")
    return()
  endif()

  iree_arch_to_llvm_arch(_LLVM_ARCH "${_RULE_ARCH}")

  set(_COPTS
    # Target architecture.
    "-target" "${_LLVM_ARCH}"

    # C17 with no system deps.
    "-std=c17"
    "-nostdinc"
    "-ffreestanding"

    # Optimized and unstamped.
    "-O3"
    "-DNDEBUG"
    "-fno-ident"
    "-fdiscard-value-names"

    # Set the size of wchar_t to 4 bytes (instead of 2 bytes).
    # This must match what the runtime is built with.
    "-fno-short-wchar"

    # Enable inline asm.
    "-fasm"

    # Object file only in bitcode format.
    "-c"
    "-emit-llvm"

    # Force the library into standalone mode (not depending on build-directory
    # configuration).
    "-DIREE_DEVICE_STANDALONE=1"
  )

  list(APPEND _COPTS "-isystem" "${IREE_CLANG_BUILTIN_HEADERS_PATH}")
  list(APPEND _COPTS "-I" "${IREE_SOURCE_DIR}/runtime/src")
  list(APPEND _COPTS "-I" "${IREE_BINARY_DIR}/runtime/src")
  list(APPEND _COPTS "${_RULE_COPTS}")

  if (_RULE_ARCH STREQUAL "arm_32")
    # Silence "warning: unknown platform, assuming -mfloat-abi=soft"
    list(APPEND _COPTS "-mfloat-abi=soft")
  elseif(_RULE_ARCH STREQUAL "riscv_32")
    # On RISC-V, linking LLVM modules requires matching target-abi.
    # https://lists.llvm.org/pipermail/llvm-dev/2020-January/138450.html
    # The choice of ilp32d is simply what we have in existing riscv_32 tests.
    # Open question - how do we scale to supporting all RISC-V ABIs?
    list(APPEND _COPTS "-mabi=ilp32d")
  elseif(_RULE_ARCH STREQUAL "riscv_64")
    # Same comments as above riscv_32 case.
    list(APPEND _COPTS "-mabi=lp64d")
  endif()

  set(_BITCODE_FILES)
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_BITCODE_SRC_PATH "${_SRC}" REALPATH)
    set(_BITCODE_FILE "${_RULE_NAME}_${_SRC}.bc")
    list(APPEND _BITCODE_FILES ${_BITCODE_FILE})
    add_custom_command(
      OUTPUT
        "${_BITCODE_FILE}"
      COMMAND
        "${IREE_CLANG_BINARY}"
        ${_COPTS}
        "${_BITCODE_SRC_PATH}"
        "-o"
        "${_BITCODE_FILE}"
      DEPENDS
        "${IREE_CLANG_BINARY}"
        "${_SRC}"
        "${_RULE_INTERNAL_HDRS}"
      COMMENT
        "Compiling ${_SRC} to ${_BITCODE_FILE}"
      VERBATIM
    )
  endforeach()

  add_custom_command(
    OUTPUT
      ${_OUT}
    COMMAND
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
      "-o"
      "${_OUT}"
    DEPENDS
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
    COMMENT
      "Linking bitcode to ${_OUT}"
    VERBATIM
  )

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()

function(iree_opencl_binary)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT;TARGET;ARCH"
    "SRCS;COPTS;LINKOPTS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.so")
  endif()

  # Using a native cmake target will let us get compile_commands.json entries
  # and make IDEs/clangd happy. It does require some special handling, though.
  # CMake compiler flags get deduped and for things like `-Xclang` that are
  # prefixes we need to sure they don't by disabling CMake munging.
  #
  # This has only been tested on ninja - other cmake generators may not be ok
  # with having CMAKE_C_COMPILER change during CMakeLists.txt processing.
  set(_USE_CMAKE_RULE OFF)
  if(CMAKE_GENERATOR STREQUAL "Ninja")
    set(_USE_CMAKE_RULE ON)
  endif()
  if(_USE_CMAKE_RULE)
    set(_COPTS_XCLANG
      "SHELL:-x cl"
      "SHELL:-Xclang -cl-std=CL2.0"
      "SHELL:-Xclang -finclude-default-header"
      "SHELL:-Xclang -fdeclare-opencl-builtins"
    )
  else()
    set(_COPTS_XCLANG
      "-x" "cl"
      "-Xclang" "-cl-std=CL2.0"
      "-Xclang" "-finclude-default-header"
      "-Xclang" "-fdeclare-opencl-builtins"
    )
  endif()

  set(_COPTS
    # OpenCL configuration.
    "${_COPTS_XCLANG}"
    "-cl-no-stdinc"
    "-nogpulib"
    "-fno-short-wchar"

    # Target architecture/machine.
    "-target" "${_RULE_TARGET}"
    "-march=${_RULE_ARCH}"
    "-fgpu-rdc"  # NOTE: may not be required for all targets

    # Header paths for builtins and our own includes.
    "-isystem" "${IREE_CLANG_BUILTIN_HEADERS_PATH}"
    "-I${IREE_SOURCE_DIR}/runtime/src"
    "-I${IREE_BINARY_DIR}/runtime/src"

    # Optimized.
    "-fno-ident"
    "-fvisibility=hidden"
    "-O3"

    # Object file only in bitcode format.
    "-c"
    "-emit-llvm"
  )

  if(_USE_CMAKE_RULE)
    set(_ORIGINAL_C_COMPILER "${CMAKE_C_COMPILER}")
    set(_ORIGINAL_C_FLAGS "${CMAKE_C_FLAGS}")
    set(_ORIGINAL_C_STANDARD "${CMAKE_C_STANDARD}")
    set(CMAKE_C_COMPILER "${IREE_CLANG_BINARY}")
    set(CMAKE_C_FLAGS)
    set(CMAKE_C_STANDARD)

    set(_BITCODE_RULE "${_PACKAGE_NAME}_${_RULE_NAME}_bc")
    add_library(${_BITCODE_RULE} STATIC)
    target_sources(${_BITCODE_RULE} PRIVATE ${_RULE_SRCS})
    target_compile_options(${_BITCODE_RULE} PRIVATE ${_COPTS})
    set_target_properties(${_BITCODE_RULE} PROPERTIES PREFIX "")
    set_target_properties(${_BITCODE_RULE} PROPERTIES SUFFIX ".a")
    set_target_properties(${_BITCODE_RULE} PROPERTIES OUTPUT_NAME ${_RULE_NAME})
    set_target_properties(${_BITCODE_RULE} PROPERTIES C_STANDARD_REQUIRED OFF)
    set_target_properties(${_BITCODE_RULE} PROPERTIES LINKER_LANGUAGE C)

    set(_ARCHIVE_FILE "${_RULE_NAME}.a")

    set(CMAKE_C_COMPILER "${_ORIGINAL_C_COMPILER}")
    set(CMAKE_C_FLAGS "${_ORIGINAL_C_FLAGS}")
    set(CMAKE_C_STANDARD "${_ORIGINAL_C_STANDARD}")
  else()
    set(_BITCODE_FILES)
    foreach(_SRC ${_RULE_SRCS})
      get_filename_component(_BITCODE_SRC_PATH "${_SRC}" REALPATH)
      set(_BITCODE_FILE "${_SRC}.bc")
      list(APPEND _BITCODE_FILES ${_BITCODE_FILE})
      add_custom_command(
        OUTPUT
          "${_BITCODE_FILE}"
        COMMAND
          "${IREE_CLANG_BINARY}"
          ${_COPTS}
          "${_BITCODE_SRC_PATH}"
          "-o"
          "${_BITCODE_FILE}"
        DEPENDS
          "${IREE_CLANG_BINARY}"
          "${_SRC}"
        COMMENT
          "Compiling ${_SRC} to ${_BITCODE_FILE}"
        VERBATIM
      )
    endforeach()

    set(_ARCHIVE_FILE "${_RULE_NAME}.a")
    add_custom_command(
      OUTPUT
        ${_ARCHIVE_FILE}
      COMMAND
        ${IREE_LLVM_LINK_BINARY}
        ${_BITCODE_FILES}
        "-o"
        "${_ARCHIVE_FILE}"
      DEPENDS
        ${IREE_LLVM_LINK_BINARY}
        ${_BITCODE_FILES}
      COMMENT
        "Archiving bitcode to ${_ARCHIVE_FILE}"
      VERBATIM
    )
  endif()

  set(_LINKED_FILE "${_RULE_NAME}.bc")
  add_custom_command(
    OUTPUT
      ${_LINKED_FILE}
    COMMAND
      ${IREE_LLVM_LINK_BINARY}
      "-internalize"
      "-only-needed"
      "${_ARCHIVE_FILE}"
      "-o" "${_LINKED_FILE}"
    DEPENDS
      "${IREE_LLVM_LINK_BINARY}"
      "${_ARCHIVE_FILE}"
    COMMENT
      "Linking bitcode to ${_LINKED_FILE}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT
      "${_OUT}"
    COMMAND ${IREE_LLD_BINARY}
      "-flavor" "gnu"
      "-m" "elf64_amdgpu"
      "--build-id=none"
      "--no-undefined"
      "-shared"
      "-plugin-opt=mcpu=${_RULE_ARCH}"
      "-plugin-opt=O3"
      "--lto-CGO3"
      "--no-whole-archive"
      "--gc-sections"
      "--strip-all"
      "${_LINKED_FILE}"
      "-o" "${_OUT}"
    DEPENDS
      "${_LINKED_FILE}"
      "${IREE_LLD_TARGET}"
    COMMENT
      "Compiling binary to ${_OUT}"
    VERBATIM
  )

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()

function(iree_cuda_bitcode_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT;CUDA_ARCH"
    "SRCS;COPTS"
    ${ARGN}
  )

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.bc")
  endif()

  set(_CUDA_ARCH "${_RULE_CUDA_ARCH}")

  set(_COPTS
    "-x" "cuda"

    # Target architecture.
    "--cuda-gpu-arch=${_CUDA_ARCH}"

    "--cuda-path=${CUDAToolkit_ROOT}"

    # Suppress warnings about missing path to cuda lib,
    # and benign warning about CUDA version.
    "-Wno-unknown-cuda-version"
    "-nocudalib"
    "--cuda-device-only"

    # https://github.com/llvm/llvm-project/issues/54609
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

    # Optimized and unstamped.
    "-O3"

    # Object file only in bitcode format.
    "-c"
    "-emit-llvm"
  )

  set(_BITCODE_FILES)
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_BITCODE_SRC_PATH "${_SRC}" REALPATH)
    set(_BITCODE_FILE "${_RULE_NAME}_${_SRC}.bc")
    list(APPEND _BITCODE_FILES ${_BITCODE_FILE})
    add_custom_command(
      OUTPUT
        "${_BITCODE_FILE}"
      COMMAND
        "${IREE_CLANG_BINARY}"
        ${_COPTS}
        "${_BITCODE_SRC_PATH}"
        "-o"
        "${_BITCODE_FILE}"
      DEPENDS
        "${IREE_CLANG_BINARY}"
        "${_SRC}"
      COMMENT
        "Compiling ${_SRC} to ${_BITCODE_FILE}"
      VERBATIM
    )
  endforeach()

  add_custom_command(
    OUTPUT
      ${_OUT}
    COMMAND
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
      "-o"
      "${_OUT}"
    DEPENDS
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
    COMMENT
      "Linking bitcode to ${_OUT}"
    VERBATIM
  )

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()

# iree_link_bitcode()
#
# Builds an LLVM bitcode library from an input file via clang
#
# Parameters:
# NAME: Name of target (see Note).
# SRCS: Source files to pass to clang.
# OUT: Output file name (defaults to NAME.bc).
function(iree_link_bitcode)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT"
    "SRCS"
    ${ARGN}
  )

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.bc")
  endif()

  set(_BITCODE_FILES "${_RULE_SRCS}")

  add_custom_command(
    OUTPUT
      ${_OUT}
    COMMAND
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
      "-o"
      "${_OUT}"
    DEPENDS
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
    COMMENT
      "Linking bitcode to ${_OUT}"
    VERBATIM
  )

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()
