// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_ref_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): simplify this for command buffers by pre-baking as much as we
// can during the queue issue - we can at least dereference handles and add in
// the offset for everything such that we only have to deal with the slot offset
// and have less branchy code.
IREE_OCL_GLOBAL void* iree_hal_amdgpu_device_buffer_ref_resolve(
    iree_hal_amdgpu_device_buffer_ref_t buffer_ref,
    IREE_OCL_ALIGNAS(64)
        const iree_hal_amdgpu_device_buffer_ref_t* IREE_OCL_RESTRICT
            binding_table) {
  if (iree_hal_amdgpu_device_buffer_ref_type(buffer_ref) ==
      IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT) {
    const iree_hal_amdgpu_device_buffer_ref_t binding =
        binding_table[buffer_ref.value.slot];
    const uint64_t offset =
        iree_hal_amdgpu_device_buffer_ref_offset(buffer_ref) +
        iree_hal_amdgpu_device_buffer_ref_offset(binding);
    const uint64_t length =
        iree_hal_amdgpu_device_buffer_ref_length(buffer_ref) == UINT64_MAX
            ? iree_hal_amdgpu_device_buffer_ref_length(binding) -
                  iree_hal_amdgpu_device_buffer_ref_offset(buffer_ref)
            : iree_hal_amdgpu_device_buffer_ref_length(buffer_ref);
    iree_hal_amdgpu_device_buffer_ref_set(
        buffer_ref, iree_hal_amdgpu_device_buffer_ref_type(binding), offset,
        length, binding.value.bits);
  }
  if (iree_hal_amdgpu_device_buffer_ref_type(buffer_ref) ==
      IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE) {
    buffer_ref.value.ptr = buffer_ref.value.handle->ptr;
  }
  return buffer_ref.value.ptr
             ? (IREE_OCL_GLOBAL uint8_t*)buffer_ref.value.ptr +
                   iree_hal_amdgpu_device_buffer_ref_offset(buffer_ref)
             : NULL;
}

IREE_OCL_GLOBAL void* iree_hal_amdgpu_device_workgroup_count_buffer_ref_resolve(
    iree_hal_amdgpu_device_workgroup_count_buffer_ref_t buffer_ref,
    IREE_OCL_ALIGNAS(64)
        const iree_hal_amdgpu_device_buffer_ref_t* IREE_OCL_RESTRICT
            binding_table) {
  if (iree_hal_amdgpu_device_workgroup_count_buffer_ref_type(buffer_ref) ==
      IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_SLOT) {
    const iree_hal_amdgpu_device_buffer_ref_t binding =
        binding_table[buffer_ref.value.slot];
    const uint64_t offset =
        iree_hal_amdgpu_device_workgroup_count_buffer_ref_offset(buffer_ref) +
        iree_hal_amdgpu_device_buffer_ref_offset(binding);
    iree_hal_amdgpu_device_workgroup_count_buffer_ref_set(
        buffer_ref, iree_hal_amdgpu_device_buffer_ref_type(binding), offset,
        binding.value.bits);
  }
  if (iree_hal_amdgpu_device_workgroup_count_buffer_ref_type(buffer_ref) ==
      IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE) {
    buffer_ref.value.ptr = buffer_ref.value.handle->ptr;
  }
  return buffer_ref.value.ptr
             ? (IREE_OCL_GLOBAL uint8_t*)buffer_ref.value.ptr +
                   iree_hal_amdgpu_device_workgroup_count_buffer_ref_offset(
                       buffer_ref)
             : NULL;
}

//===----------------------------------------------------------------------===//
// Buffer transfer operation utilities
//===----------------------------------------------------------------------===//

// Reserves the next packet in the queue and returns its packet_id.
// If tracing is enabled |out_completion_signal| will be populated with the
// signal that must be attached to the operation.
static uint64_t iree_hal_amdgpu_device_buffer_op_reserve(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_buffer_transfer_state_t*
        IREE_OCL_RESTRICT state,
    iree_hal_amdgpu_trace_execution_zone_type_t zone_type,
    IREE_OCL_PRIVATE iree_hsa_signal_t* IREE_OCL_RESTRICT
        out_completion_signal) {
#if IREE_HAL_AMDGPU_TRACING_FEATURES & \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION
  if (state->trace_buffer) {
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id =
        iree_hal_amdgpu_device_query_ringbuffer_acquire(
            &state->trace_buffer->query_ringbuffer);
    *out_completion_signal =
        iree_hal_amdgpu_device_trace_execution_zone_dispatch(
            state->trace_buffer, zone_type, 0, execution_query_id);
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION

  // Reserve the next packet in the queue.
  const uint64_t packet_id = iree_hsa_queue_add_write_index(
      state->queue, 1u, iree_hal_amdgpu_device_memory_order_relaxed);
  while (packet_id -
             iree_hsa_queue_load_read_index(
                 state->queue, iree_hal_amdgpu_device_memory_order_acquire) >=
         state->queue->size) {
    iree_hal_amdgpu_device_yield();  // spinning
  }

  return packet_id;
}

// Commits a reserved transfer packet.
// The header will be updated and the target queue doorbell will be signaled.
static void iree_hal_amdgpu_device_buffer_op_commit(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_buffer_transfer_state_t*
        IREE_OCL_RESTRICT state,
    uint64_t packet_id,
    IREE_OCL_GLOBAL iree_hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT packet,
    iree_hsa_signal_t completion_signal) {
  // Chain completion.
  packet->completion_signal = completion_signal;

  // Populate the header and release the packet to the queue.
  uint16_t header = IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH
                    << IREE_HSA_PACKET_HEADER_TYPE;

  // NOTE: we don't need a barrier bit as the caller is expecting it to run
  // concurrently if needed.
  header |= 0 << IREE_HSA_PACKET_HEADER_BARRIER;

#if IREE_HAL_AMDGPU_TRACING_FEATURES & \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION
  if (state->trace_buffer) {
    // Force a barrier bit if we are tracing execution. This ensures that we get
    // exclusive timing for the operation.
    header |= 1 << IREE_HSA_PACKET_HEADER_BARRIER;
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION

  // TODO(benvanik): scope to agent if the pointer is local, or maybe none in
  // cases where surrounding barriers performed the cache management.
  header |= IREE_HSA_FENCE_SCOPE_SYSTEM
            << IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  header |= IREE_HSA_FENCE_SCOPE_SYSTEM
            << IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;

  const uint32_t header_setup = header | (uint32_t)(packet->setup << 16);
  iree_hal_amdgpu_device_atomic_store_explicit(
      (IREE_OCL_GLOBAL iree_hal_amdgpu_device_atomic_uint32_t*)packet,
      header_setup, iree_hal_amdgpu_device_memory_order_release,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);

  // Signal the queue doorbell indicating the packet has been updated.
  iree_hsa_signal_store(state->queue->doorbell_signal, packet_id,
                        iree_hal_amdgpu_device_memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_fill_*
//===----------------------------------------------------------------------===//

__kernel void iree_hal_amdgpu_device_buffer_fill_x1(
    IREE_OCL_GLOBAL void* IREE_OCL_RESTRICT target_ptr, const uint64_t length,
    const uint8_t pattern) {
  // DO NOT SUBMIT
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_1byte
}

__kernel void iree_hal_amdgpu_device_buffer_fill_x2(
    IREE_OCL_GLOBAL void* IREE_OCL_RESTRICT target_ptr, const uint64_t length,
    const uint16_t pattern) {
  // DO NOT SUBMIT
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_2byte
}

__kernel void iree_hal_amdgpu_device_buffer_fill_x4(
    IREE_OCL_GLOBAL void* IREE_OCL_RESTRICT target_ptr, const uint64_t length,
    const uint32_t pattern) {
  // DO NOT SUBMIT
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_4byte
}

__kernel void iree_hal_amdgpu_device_buffer_fill_x8(
    IREE_OCL_GLOBAL void* IREE_OCL_RESTRICT target_ptr, const uint64_t length,
    const uint64_t pattern) {
  // DO NOT SUBMIT
  // runtime/src/iree/hal/drivers/metal/builtin/fill_buffer_generic.metal
  // fill_buffer_8byte
}

IREE_OCL_GLOBAL iree_hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT
iree_hal_amdgpu_device_buffer_fill_emplace_reserve(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_buffer_transfer_state_t*
        IREE_OCL_RESTRICT state,
    IREE_OCL_GLOBAL void* target_ptr, const uint64_t length,
    const uint64_t pattern, const uint8_t pattern_length,
    IREE_OCL_GLOBAL uint64_t* IREE_OCL_RESTRICT kernarg_ptr,
    const uint64_t packet_id) {
  IREE_OCL_TRACE_BUFFER_DEFINE(state->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);

  // Update kernargs (same for all kernels).
  kernarg_ptr[0] = (uint64_t)target_ptr;
  kernarg_ptr[1] = length;
  kernarg_ptr[2] = pattern;

  // Select the kernel for the fill operation.
  IREE_OCL_GLOBAL const iree_hal_amdgpu_device_kernel_args_t* IREE_OCL_RESTRICT
      kernel_args = NULL;
  uint64_t block_size = 0;
  switch (pattern_length) {
    case 1:
      IREE_OCL_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x1");
      kernel_args = &state->kernels->blit.fill_x1;
      block_size = 1;
      break;
    case 2:
      IREE_OCL_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x2");
      kernel_args = &state->kernels->blit.fill_x2;
      block_size = 1;
      break;
    case 4:
      IREE_OCL_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x4");
      kernel_args = &state->kernels->blit.fill_x4;
      block_size = 1;
      break;
    case 8:
      IREE_OCL_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "fill_x8");
      kernel_args = &state->kernels->blit.fill_x8;
      block_size = 1;
      break;
  }
  IREE_OCL_TRACE_ZONE_APPEND_VALUE_I64(z0, block_size);

  // Populate the packet.
  const uint64_t queue_mask = state->queue->size - 1;  // power of two
  IREE_OCL_GLOBAL iree_hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT
      dispatch_packet =
          state->queue->base_address + (packet_id & queue_mask) * 64;
  dispatch_packet->setup = kernel_args->setup;
  dispatch_packet->workgroup_size[0] = kernel_args->workgroup_size[0];
  dispatch_packet->workgroup_size[1] = kernel_args->workgroup_size[1];
  dispatch_packet->workgroup_size[2] = kernel_args->workgroup_size[2];
  dispatch_packet->reserved0 = 0;
  dispatch_packet->grid_size[0] = 0;  // DO NOT SUBMIT block count?
  dispatch_packet->grid_size[1] = 1;
  dispatch_packet->grid_size[2] = 1;
  dispatch_packet->private_segment_size = kernel_args->private_segment_size;
  dispatch_packet->group_segment_size = kernel_args->group_segment_size;
  dispatch_packet->kernel_object = kernel_args->kernel_object;
  dispatch_packet->kernarg_address = kernarg_ptr;
  dispatch_packet->reserved2 = 0;

  IREE_OCL_TRACE_ZONE_END(z0);
  return dispatch_packet;
}

void iree_hal_amdgpu_device_buffer_fill_enqueue(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_buffer_transfer_state_t*
        IREE_OCL_RESTRICT state,
    IREE_OCL_GLOBAL void* target_ptr, const uint64_t length,
    const uint64_t pattern, const uint8_t pattern_length,
    IREE_OCL_GLOBAL uint64_t* IREE_OCL_RESTRICT kernarg_ptr) {
  IREE_OCL_TRACE_BUFFER_DEFINE(state->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);
  IREE_OCL_TRACE_ZONE_APPEND_VALUE_I64(z0, length);

  // Reserve and begin populating the operation packet.
  // When tracing is enabled capture the timing signal.
  iree_hsa_signal_t completion_signal = iree_hsa_signal_null();
  const uint64_t packet_id = iree_hal_amdgpu_device_buffer_op_reserve(
      state, IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_FILL,
      &completion_signal);

  // Emplace the dispatch packet into the queue.
  // Note that until the packet is issued the queue will stall.
  IREE_OCL_GLOBAL iree_hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT packet =
      iree_hal_amdgpu_device_buffer_fill_emplace_reserve(
          state, target_ptr, length, pattern, pattern_length, kernarg_ptr,
          packet_id);

  // Issues the buffer operation packet by configuring its header and signaling
  // the queue doorbell.
  iree_hal_amdgpu_device_buffer_op_commit(state, packet_id, packet,
                                          completion_signal);

  IREE_OCL_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_buffer_copy_*
//===----------------------------------------------------------------------===//

__kernel void iree_hal_amdgpu_device_buffer_copy_x1(
    IREE_OCL_GLOBAL const uint8_t* IREE_OCL_RESTRICT source_ptr,
    IREE_OCL_GLOBAL uint8_t* IREE_OCL_RESTRICT target_ptr,
    const uint64_t length) {
  // DO NOT SUBMIT
  // runtime/src/iree/hal/drivers/metal/builtin/copy_buffer_generic.metal
  // copy_buffer_1byte
}

__kernel void iree_hal_amdgpu_device_buffer_copy_x2(
    IREE_OCL_GLOBAL const uint16_t* IREE_OCL_RESTRICT source_ptr,
    IREE_OCL_GLOBAL uint16_t* IREE_OCL_RESTRICT target_ptr,
    const uint64_t length) {
  // DO NOT SUBMIT
}

__kernel void iree_hal_amdgpu_device_buffer_copy_x4(
    IREE_OCL_GLOBAL const uint16_t* IREE_OCL_RESTRICT source_ptr,
    IREE_OCL_GLOBAL uint16_t* IREE_OCL_RESTRICT target_ptr,
    const uint64_t length) {
  // DO NOT SUBMIT
}

__kernel void iree_hal_amdgpu_device_buffer_copy_x8(
    IREE_OCL_GLOBAL const uint16_t* IREE_OCL_RESTRICT source_ptr,
    IREE_OCL_GLOBAL uint16_t* IREE_OCL_RESTRICT target_ptr,
    const uint64_t length) {
  // DO NOT SUBMIT
}

// TODO(benvanik): experiment with best widths for bulk transfers.
__kernel void iree_hal_amdgpu_device_buffer_copy_x64(
    IREE_OCL_GLOBAL const uint16_t* IREE_OCL_RESTRICT source_ptr,
    IREE_OCL_GLOBAL uint16_t* IREE_OCL_RESTRICT target_ptr,
    const uint64_t length) {
  // DO NOT SUBMIT
}

// TODO(benvanik): experiment with enqueuing SDMA somehow (may need to take a
// DMA queue as well as the dispatch queue).
IREE_OCL_GLOBAL iree_hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT
iree_hal_amdgpu_device_buffer_copy_emplace_reserve(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_buffer_transfer_state_t*
        IREE_OCL_RESTRICT state,
    IREE_OCL_GLOBAL const void* source_ptr, IREE_OCL_GLOBAL void* target_ptr,
    const uint64_t length,
    IREE_OCL_GLOBAL uint64_t* IREE_OCL_RESTRICT kernarg_ptr,
    const uint64_t packet_id) {
  IREE_OCL_TRACE_BUFFER_DEFINE(state->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);

  // Update kernargs (same for all kernels).
  kernarg_ptr[0] = (uint64_t)source_ptr;
  kernarg_ptr[1] = (uint64_t)target_ptr;
  kernarg_ptr[2] = length;

  // Select the kernel for the copy operation.
  // TODO(benvanik): switch kernel based on source/target/length alignment.
  const iree_hal_amdgpu_device_kernel_args_t kernel_args =
      state->kernels->blit.copy_x1;
  const uint64_t block_size = 128;
  IREE_OCL_TRACE_ZONE_APPEND_TEXT_LITERAL(z0, "copy_x1");
  IREE_OCL_TRACE_ZONE_APPEND_VALUE_I64(z0, block_size);

  // Populate the packet.
  const uint64_t queue_mask = state->queue->size - 1;  // power of two
  IREE_OCL_GLOBAL iree_hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT
      dispatch_packet =
          state->queue->base_address + (packet_id & queue_mask) * 64;
  dispatch_packet->setup = kernel_args.setup;
  dispatch_packet->workgroup_size[0] = kernel_args.workgroup_size[0];
  dispatch_packet->workgroup_size[1] = kernel_args.workgroup_size[1];
  dispatch_packet->workgroup_size[2] = kernel_args.workgroup_size[2];
  dispatch_packet->reserved0 = 0;
  dispatch_packet->grid_size[0] = 0;  // DO NOT SUBMIT block count?
  dispatch_packet->grid_size[1] = 1;
  dispatch_packet->grid_size[2] = 1;
  dispatch_packet->private_segment_size = kernel_args.private_segment_size;
  dispatch_packet->group_segment_size = kernel_args.group_segment_size;
  dispatch_packet->kernel_object = kernel_args.kernel_object;
  dispatch_packet->kernarg_address = kernarg_ptr;
  dispatch_packet->reserved2 = 0;

  IREE_OCL_TRACE_ZONE_END(z0);
  return dispatch_packet;
}

void iree_hal_amdgpu_device_buffer_copy_enqueue(
    IREE_OCL_GLOBAL const iree_hal_amdgpu_device_buffer_transfer_state_t*
        IREE_OCL_RESTRICT state,
    IREE_OCL_GLOBAL const void* source_ptr, IREE_OCL_GLOBAL void* target_ptr,
    const uint64_t length,
    IREE_OCL_GLOBAL uint64_t* IREE_OCL_RESTRICT kernarg_ptr) {
  IREE_OCL_TRACE_BUFFER_DEFINE(state->trace_buffer);
  IREE_OCL_TRACE_ZONE_BEGIN(z0);
  IREE_OCL_TRACE_ZONE_APPEND_VALUE_I64(z0, length);

  // Reserve and begin populating the operation packet.
  // When tracing is enabled capture the timing signal.
  iree_hsa_signal_t completion_signal = iree_hsa_signal_null();
  const uint64_t packet_id = iree_hal_amdgpu_device_buffer_op_reserve(
      state, IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_COPY,
      &completion_signal);

  // Emplace the dispatch packet into the queue.
  // Note that until the packet is issued the queue will stall.
  IREE_OCL_GLOBAL iree_hsa_kernel_dispatch_packet_t* IREE_OCL_RESTRICT packet =
      iree_hal_amdgpu_device_buffer_copy_emplace_reserve(
          state, source_ptr, target_ptr, length, kernarg_ptr, packet_id);

  // Issues the buffer operation packet by configuring its header and signaling
  // the queue doorbell.
  iree_hal_amdgpu_device_buffer_op_commit(state, packet_id, packet,
                                          completion_signal);

  IREE_OCL_TRACE_ZONE_END(z0);
}
