// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_BUFFER_H_

#include "iree/hal/drivers/amdgpu/device/kernel.h"
#include "iree/hal/drivers/amdgpu/device/support/opencl.h"
#include "iree/hal/drivers/amdgpu/device/support/queue.h"
#include "iree/hal/drivers/amdgpu/device/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_allocation_handle_t
//===----------------------------------------------------------------------===//

// DO NOT SUBMIT
// host side allocates (or pools) these and iree_hal_buffer_t refs them
// host free of hal buffer would enqueue device dealloca
// can have size
// union struct for pool storage (bucket base, etc)
typedef struct iree_hal_amdgpu_device_allocation_handle_s {
  IREE_OCL_GLOBAL void* ptr;
  // pool it was allocated from?
  // block it was allocated from?
} iree_hal_amdgpu_device_allocation_handle_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_ref_t
//===----------------------------------------------------------------------===//

// Identifies the type of a buffer reference and how it should be resolved.
typedef uint8_t iree_hal_amdgpu_device_buffer_type_t;
enum iree_hal_amdgpu_device_buffer_type_e {
  // Reference is to an absolute device pointer that can be directly accessed.
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_PTR = 0u,
  // Reference is to a queue-ordered allocation handle that is only valid at
  // the time the buffer is committed. The handle will be valid for the lifetime
  // of the logical buffer and any resources referencing it but the pointer must
  // only be resolved between a corresponding alloca/dealloca.
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE,
  // Reference is to a slot in the binding table provided during execution.
  // Only one indirection is allowed (table slots cannot reference other slots
  // - yet).
  IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT,
};

// The ordinal of a slot in the binding table.
typedef uint32_t iree_hal_amdgpu_device_buffer_ordinal_t;

// Describes a subrange of a buffer that can be bound to a binding slot.
typedef struct iree_hal_amdgpu_device_buffer_ref_s {
  // Offset, in bytes, into the buffer that the binding starts at.
  // This will be added to the offset specified on each usage of the slot.
  uint64_t offset;
  // Length, in bytes, of the buffer that is available to the executable.
  // Lower 2 bits are iree_hal_amdgpu_device_buffer_type_t.
  // If OpenCL supported bitfields:
  //   uint64_t type : 2;
  //   uint64_t length : 62;
  uint64_t length_type;
  union {
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_PTR: device pointer.
    IREE_OCL_GLOBAL void* ptr;
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE: queue-ordered allocation
    // handle.
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_allocation_handle_t* handle;
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT: binding table slot.
    iree_hal_amdgpu_device_buffer_ordinal_t slot;
    // Used for setting the value.
    uint64_t bits;
  } value;
} iree_hal_amdgpu_device_buffer_ref_t;
IREE_OCL_STATIC_ASSERT(sizeof(iree_hal_amdgpu_device_buffer_ref_t) == 24,
                       "binding table entries should be 8 byte aligned");

#define iree_hal_amdgpu_device_buffer_ref_offset(buffer_ref) (buffer_ref).offset
#define iree_hal_amdgpu_device_buffer_ref_type(buffer_ref) \
  (iree_hal_amdgpu_device_buffer_type_t)((buffer_ref).length_type & 0x3)
#define iree_hal_amdgpu_device_buffer_ref_length(buffer_ref) \
  ((buffer_ref).length_type >> 2)
#define iree_hal_amdgpu_device_buffer_ref_set(buffer_ref_, type_, offset_, \
                                              length_, value_)             \
  (buffer_ref_).offset = (offset_);                                        \
  (buffer_ref_).length_type = ((length_) << 2) | (type_);                  \
  (buffer_ref_).value.bits = (uint64_t)(value_);

// Describes a buffer binding that contains a uint32_t[3] XYZ workgroup count.
// This is a size-optimized version of iree_hal_amdgpu_device_buffer_ref_t so
// that it will fit in our tiny packets. We know the length is a constant 12 and
// only need the offset, type, and value.
typedef struct iree_hal_amdgpu_device_workgroup_count_buffer_ref_s {
  // Offset, in bytes, into the buffer that the binding starts at.
  // This will be added to the offset specified on each usage of the slot.
  // Lower 2 bits are iree_hal_amdgpu_device_buffer_type_t.
  // If OpenCL supported bitfields:
  //   uint64_t type : 2;
  //   uint64_t offset : 62;
  uint64_t offset_type;
  union {
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_PTR: raw device pointer.
    IREE_OCL_GLOBAL void* ptr;
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE: queue-ordered allocation
    // handle.
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_allocation_handle_t* handle;
    // IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT: binding table slot.
    iree_hal_amdgpu_device_buffer_ordinal_t slot;
    // Used for setting the value.
    uint64_t bits;
  } value;
} iree_hal_amdgpu_device_workgroup_count_buffer_ref_t;
IREE_OCL_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_device_workgroup_count_buffer_ref_t) == 16,
    "binding table entries should be 8 byte aligned and tiny");

#define iree_hal_amdgpu_device_workgroup_count_buffer_ref_offset(buffer_ref) \
  ((buffer_ref).offset_type >> 2)
#define iree_hal_amdgpu_device_workgroup_count_buffer_ref_type(buffer_ref) \
  (iree_hal_amdgpu_device_buffer_type_t)((buffer_ref).offset_type & 0x3)
#define iree_hal_amdgpu_device_workgroup_count_buffer_ref_length(buffer_ref) \
  (sizeof(uint32_t) * 3)
#define iree_hal_amdgpu_device_workgroup_count_buffer_ref_set( \
    buffer_ref_, type_, offset_, value_)                       \
  (buffer_ref_).offset_type = ((offset_) << 2) | (type_);      \
  (buffer_ref_).value.bits = (uint64_t)(value_);

#if defined(IREE_OCL_TARGET_DEVICE)

// Resolves a buffer reference to an absolute device pointer.
// Expects that the binding table is provided if needed and has sufficient
// capacity for any slot that may be referenced. All queue-ordered allocations
// that may be provided via allocation handles must be committed prior to
// attempting to resolve them and must remain committed until all commands using
// the returned device pointer have completed.
IREE_OCL_GLOBAL void* iree_hal_amdgpu_device_buffer_ref_resolve(
    iree_hal_amdgpu_device_buffer_ref_t buffer_ref,
    IREE_OCL_ALIGNAS(64)
        const iree_hal_amdgpu_device_buffer_ref_t* IREE_OCL_RESTRICT
            binding_table);

// Resolves a workgroup count buffer reference to an absolute device pointer.
// This is equivalent to iree_hal_amdgpu_device_buffer_ref_resolve but for a
// fixed-size uint32_t[3] value. The returned pointer should have 4-byte
// alignment.
IREE_OCL_GLOBAL void* iree_hal_amdgpu_device_workgroup_count_buffer_ref_resolve(
    iree_hal_amdgpu_device_workgroup_count_buffer_ref_t buffer_ref,
    IREE_OCL_ALIGNAS(64)
        const iree_hal_amdgpu_device_buffer_ref_t* IREE_OCL_RESTRICT
            binding_table);

#endif  // IREE_OCL_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// Blit Kernels
//===----------------------------------------------------------------------===//

#define IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE (3 * sizeof(void*))
#define IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE (3 * sizeof(void*))

// State used when scheduling transfer commands.
typedef struct iree_hal_amdgpu_device_buffer_transfer_state_s {
  // Target queue that will execute the transfer operation.
  IREE_OCL_GLOBAL iree_hsa_queue_t* IREE_OCL_RESTRICT queue;
  // Handles to opaque kernel objects used to dispatch builtin kernels.
  IREE_OCL_GLOBAL const iree_hal_amdgpu_device_kernels_t* kernels;
  // Optional trace buffer used when tracing infrastructure is available.
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* trace_buffer;
} iree_hal_amdgpu_device_buffer_transfer_state_t;

#if defined(IREE_OCL_TARGET_DEVICE)

// Emplaces a fill dispatch packet in the target queue at the given index.
// The queue doorbell will not be signaled.
//
// NOTE: this only works with blits today. SDMA will require a different
// signature.
IREE_OCL_GLOBAL iree_hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT
iree_hal_amdgpu_device_buffer_fill_emplace_reserve(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_buffer_transfer_state_t*
        IREE_OCL_RESTRICT state,
    IREE_OCL_GLOBAL void* target_ptr, const uint64_t length,
    const uint64_t pattern, const uint8_t pattern_length,
    IREE_OCL_GLOBAL uint64_t* IREE_OCL_RESTRICT kernarg_ptr,
    const uint64_t packet_id);

// Enqueues a fill dispatch packet in the target queue.
// The packet will be acquired at the current write_index and the queue doorbell
// will be signaled.
void iree_hal_amdgpu_device_buffer_fill_enqueue(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_buffer_transfer_state_t*
        IREE_OCL_RESTRICT state,
    IREE_OCL_GLOBAL void* target_ptr, const uint64_t length,
    const uint64_t pattern, const uint8_t pattern_length,
    IREE_OCL_GLOBAL uint64_t* IREE_OCL_RESTRICT kernarg_ptr);

// Emplaces a copy dispatch packet in the target queue at the given index.
// The queue doorbell will not be signaled.
//
// NOTE: this only works with blits today. SDMA will require a different
// signature.
IREE_OCL_GLOBAL iree_hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT
iree_hal_amdgpu_device_buffer_copy_emplace_reserve(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_buffer_transfer_state_t*
        IREE_OCL_RESTRICT state,
    IREE_OCL_GLOBAL const void* source_ptr, IREE_OCL_GLOBAL void* target_ptr,
    const uint64_t length,
    IREE_OCL_GLOBAL uint64_t* IREE_OCL_RESTRICT kernarg_ptr,
    const uint64_t packet_id);

// Enqueues a copy dispatch packet in the target queue.
// The packet will be acquired at the current write_index and the queue doorbell
// will be signaled.
void iree_hal_amdgpu_device_buffer_copy_enqueue(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_buffer_transfer_state_t*
        IREE_OCL_RESTRICT state,
    IREE_OCL_GLOBAL const void* source_ptr, IREE_OCL_GLOBAL void* target_ptr,
    const uint64_t length,
    IREE_OCL_GLOBAL uint64_t* IREE_OCL_RESTRICT kernarg_ptr);

#endif  // IREE_OCL_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_BUFFER_H_
