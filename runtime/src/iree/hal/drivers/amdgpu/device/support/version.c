// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/opencl.h"

//===----------------------------------------------------------------------===//
// AMDGPU Device Library Configuration
//===----------------------------------------------------------------------===//
//
// These are normally sourced from the AMD bitcode libraries. To avoid the extra
// dependencies for what amounts to a few bools we inline them here. Note that
// these may not match our compiler-generated binaries and that's ok: here we're
// just running basic integer code for scheduling and floating point modes don't
// really matter.
//
// Sources:
// https://github.com/ROCm/rocMLIR/blob/develop/external/llvm-project/amd/device-libs/README.md

const IREE_OCL_CONSTANT bool __oclc_unsafe_math_opt = false;
const IREE_OCL_CONSTANT bool __oclc_daz_opt = false;
const IREE_OCL_CONSTANT bool __oclc_correctly_rounded_sqrt32 = true;
const IREE_OCL_CONSTANT bool __oclc_finite_only_opt = false;
const IREE_OCL_CONSTANT bool __oclc_wavefrontsize64 =
    __AMDGCN_WAVEFRONT_SIZE__ == 64 ? 1 : 0;

#if !defined(IREE_OCL_ISA_VERSION)
#if defined(__gfx700__)
#define IREE_OCL_ISA_VERSION 7000
#elif defined(__gfx701__)
#define IREE_OCL_ISA_VERSION 7001
#elif defined(__gfx702__)
#define IREE_OCL_ISA_VERSION 7002
#elif defined(__gfx703__)
#define IREE_OCL_ISA_VERSION 7003
#elif defined(__gfx704__)
#define IREE_OCL_ISA_VERSION 7004
#elif defined(__gfx705__)
#define IREE_OCL_ISA_VERSION 7005
#elif defined(__gfx801__)
#define IREE_OCL_ISA_VERSION 8001
#elif defined(__gfx802__)
#define IREE_OCL_ISA_VERSION 8002
#elif defined(__gfx803__)
#define IREE_OCL_ISA_VERSION 8003
#elif defined(__gfx805__)
#define IREE_OCL_ISA_VERSION 8005
#elif defined(__gfx810__)
#define IREE_OCL_ISA_VERSION 8100
#elif defined(__gfx900__)
#define IREE_OCL_ISA_VERSION 9000
#elif defined(__gfx902__)
#define IREE_OCL_ISA_VERSION 9002
#elif defined(__gfx904__)
#define IREE_OCL_ISA_VERSION 9004
#elif defined(__gfx906__)
#define IREE_OCL_ISA_VERSION 9006
#elif defined(__gfx908__)
#define IREE_OCL_ISA_VERSION 9008
#elif defined(__gfx909__)
#define IREE_OCL_ISA_VERSION 9009
#elif defined(__gfx90a__)
#define IREE_OCL_ISA_VERSION 9010
#elif defined(__gfx90c__)
#define IREE_OCL_ISA_VERSION 9012
#elif defined(__gfx940__)
#define IREE_OCL_ISA_VERSION 9400
#elif defined(__gfx941__)
#define IREE_OCL_ISA_VERSION 9401
#elif defined(__gfx942__)
#define IREE_OCL_ISA_VERSION 9402
#elif defined(__gfx1010__)
#define IREE_OCL_ISA_VERSION 10100
#elif defined(__gfx1011__)
#define IREE_OCL_ISA_VERSION 10101
#elif defined(__gfx1012__)
#define IREE_OCL_ISA_VERSION 10102
#elif defined(__gfx1013__)
#define IREE_OCL_ISA_VERSION 10103
#elif defined(__gfx1030__)
#define IREE_OCL_ISA_VERSION 10300
#elif defined(__gfx1031__)
#define IREE_OCL_ISA_VERSION 10301
#elif defined(__gfx1032__)
#define IREE_OCL_ISA_VERSION 10302
#elif defined(__gfx1033__)
#define IREE_OCL_ISA_VERSION 10303
#elif defined(__gfx1034__)
#define IREE_OCL_ISA_VERSION 10304
#elif defined(__gfx1035__)
#define IREE_OCL_ISA_VERSION 10305
#elif defined(__gfx1036__)
#define IREE_OCL_ISA_VERSION 10306
#elif defined(__gfx1100__)
#define IREE_OCL_ISA_VERSION 11000
#elif defined(__gfx1101__)
#define IREE_OCL_ISA_VERSION 11001
#elif defined(__gfx1102__)
#define IREE_OCL_ISA_VERSION 11002
#elif defined(__gfx1103__)
#define IREE_OCL_ISA_VERSION 11003
#elif defined(__gfx1150__)
#define IREE_OCL_ISA_VERSION 11500
#elif defined(__gfx1151__)
#define IREE_OCL_ISA_VERSION 11501
#elif defined(__gfx1200__)
#define IREE_OCL_ISA_VERSION 12000
#elif defined(__gfx1201__)
#define IREE_OCL_ISA_VERSION 12001
#else
// NOTE: if you're seeing this then it's likely that you need to add a new
// elif to this region.
#error "unknown AMDGPU arch; use -DIREE_OCL_ISA_VERSION= to override"
#endif  // __gfxNNNN___
#endif  // IREE_OCL_ISA_VERSION

// Normally sourced from amdgcn/bitcode/oclc/oclc_isa_version_NNNN.bc.
const IREE_OCL_CONSTANT unsigned __oclc_ISA_version = IREE_OCL_ISA_VERSION;
