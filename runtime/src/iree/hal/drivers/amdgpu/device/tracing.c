// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_query_ringbuffer_t
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_device_query_ringbuffer_initialize(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_query_ringbuffer_t* IREE_OCL_RESTRICT
        out_ringbuffer) {
  // NOTE: we don't memset here as it should have been zeroed already.
  for (uint32_t i = 0; i < IREE_OCL_ARRAYSIZE(out_ringbuffer->signals); ++i) {
    out_ringbuffer->signals[i].kind = IREE_AMD_SIGNAL_KIND_USER;
    out_ringbuffer->signals[i].value = 1;
  }
}

iree_hal_amdgpu_trace_execution_query_id_t
iree_hal_amdgpu_device_query_ringbuffer_acquire(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_query_ringbuffer_t* IREE_OCL_RESTRICT
        ringbuffer) {
  // Slice off a single value and return it mapped into a query ID.
  return (ringbuffer->write_index++) &
         (IREE_OCL_ARRAYSIZE(ringbuffer->signals) - 1);
}

uint64_t iree_hal_amdgpu_device_query_ringbuffer_acquire_range(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_query_ringbuffer_t* IREE_OCL_RESTRICT
        ringbuffer,
    uint16_t count) {
  // Slice off another chunk.
  uint64_t base_index = ringbuffer->write_index;
  ringbuffer->write_index += count;
  return base_index;
}

void iree_hal_amdgpu_device_query_ringbuffer_release_range(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_query_ringbuffer_t* IREE_OCL_RESTRICT
        ringbuffer,
    uint16_t count) {
  // Reset all returned signals.
  for (uint32_t i = ringbuffer->read_index; i < ringbuffer->read_index + count;
       ++i) {
    IREE_OCL_GLOBAL iree_amd_signal_t* signal =
        &ringbuffer->signals[i & (IREE_OCL_ARRAYSIZE(ringbuffer->signals) - 1)];
    signal->value = 1;
    signal->start_ts = 0;
    signal->end_ts = 0;
  }
  ringbuffer->read_index += count;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_trace_buffer_t
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_TRACING_FEATURES

// Reserves |length| bytes from the trace buffer and returns a pointer to it.
// Callers must populate the entire packet prior to calling
// iree_hal_amdgpu_device_trace_commit_range. Multiple reservations can be made
// between commits to batch the commit logic (which usually involves a host
// interrupt to flush the ringbuffer).
static inline IREE_OCL_GLOBAL void* iree_hal_amdgpu_device_trace_reserve_range(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    size_t length) {
  // Reserve a range of the requested size from the current reservation offset.
  // NOTE: this is only modified on device and on the agent associated with the
  // scheduler that's calling this and as such only has to be at device scope.
  uint64_t write_offset = iree_hal_amdgpu_device_atomic_fetch_add_explicit(
      &trace_buffer->write_reserve_offset, length,
      iree_hal_amdgpu_device_memory_order_relaxed,
      iree_hal_amdgpu_device_memory_scope_device);

  // Spin until there's capacity in the ringbuffer. We need to wait until the
  // host catches up to our last flush.
  // WARNING: this may lock up forever if we really spill the ring.
  // TODO(benvanik): find a way to fail here, or throw an interrupt.
  // We could use a signal instead of an atomic but there's no good way to park
  // from the current pc.
  if (write_offset + length -
          iree_hal_amdgpu_device_atomic_load_explicit(
              &trace_buffer->read_commit_offset,
              iree_hal_amdgpu_device_memory_order_acquire,
              iree_hal_amdgpu_device_memory_scope_all_svm_devices) >
      trace_buffer->ringbuffer_capacity) {
    iree_hal_amdgpu_device_yield();
  }

  // Calculate base address of the packet within the ringbuffer. Note that it
  // may extend off the end of the base allocation but so long as the length is
  // in bounds it'll be accessing the physical memory through the subsequent
  // virtual address mapping.
  IREE_OCL_GLOBAL void* packet_ptr =
      (IREE_OCL_GLOBAL uint8_t*)trace_buffer->ringbuffer_base +
      (write_offset & iree_hal_amdgpu_device_trace_buffer_mask(trace_buffer));

  return packet_ptr;
}

bool iree_hal_amdgpu_device_trace_commit_range(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer) {
  // Bump the commit offset as seen by the host to the reserve offset at the
  // start of this call. The host may immediately begin reading from its last
  // read_commit_offset up to the new write_commit_offset and we cannot
  // overwrite any of that range until the read_commit_offset has been bumped by
  // the host.
  uint64_t last_reserve_offset = iree_hal_amdgpu_device_atomic_load_explicit(
      &trace_buffer->write_reserve_offset,
      iree_hal_amdgpu_device_memory_order_acquire,
      iree_hal_amdgpu_device_memory_scope_device);
  uint64_t last_commit_offset = iree_hal_amdgpu_device_atomic_exchange_explicit(
      &trace_buffer->write_commit_offset, last_reserve_offset,
      iree_hal_amdgpu_device_memory_order_release,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);

  // If the last commit offset matches the last reserve offset then there were
  // no pending writes to commit and the caller does not need to notify the
  // host.
  return last_reserve_offset != last_commit_offset;
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURES

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_TRACING_FEATURES & \
    IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION

iree_ocl_zone_id_t iree_hal_amdgpu_device_trace_zone_begin(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_src_loc_ptr_t src_loc) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_zone_begin_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_zone_begin_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_BEGIN;
  packet->timestamp = iree_hal_amdgpu_device_timestamp();
  packet->src_loc = src_loc;
  return 1;
}

void iree_hal_amdgpu_device_trace_zone_end(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_zone_end_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_zone_end_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_END;
  packet->timestamp = iree_hal_amdgpu_device_timestamp();
}

void iree_hal_amdgpu_device_trace_zone_append_value_i64(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    int64_t value) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_zone_value_i64_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_zone_value_i64_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_I64;
  packet->value = value;
}

void iree_hal_amdgpu_device_trace_zone_append_text_literal(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t value_literal) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_zone_value_text_literal_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer,
          sizeof(iree_hal_amdgpu_trace_zone_value_text_literal_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_LITERAL;
  packet->value = value_literal;
}

void iree_hal_amdgpu_device_trace_zone_append_text_dynamic(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    IREE_OCL_GLOBAL const char* IREE_OCL_RESTRICT value, size_t value_length) {
  const size_t total_size =
      sizeof(iree_hal_amdgpu_trace_zone_value_text_dynamic_t) + value_length;
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_zone_value_text_dynamic_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(trace_buffer, total_size);
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_DYNAMIC;
  packet->length = (uint32_t)value_length;
  iree_ocl_memcpy(packet->value, value, value_length);
}

void iree_hal_amdgpu_device_trace_plot_configure(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal,
    iree_hal_amdgpu_trace_plot_type_t type,
    iree_hal_amdgpu_trace_plot_flags_t flags,
    iree_hal_amdgpu_trace_color_t color) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_plot_config_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_plot_config_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_CONFIG;
  packet->plot_type = type;
  packet->plot_flags = flags;
  packet->color = color;
  packet->name = name_literal;
}

void iree_hal_amdgpu_device_trace_plot_value_i64(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal, int64_t value) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_plot_value_i64_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_plot_value_i64_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_VALUE_I64;
  packet->plot_name = name_literal;
  packet->timestamp = iree_hal_amdgpu_device_timestamp();
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_TRACING_FEATURES & \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

iree_hsa_signal_t iree_hal_amdgpu_device_trace_execution_zone_begin(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id,
    iree_hal_amdgpu_trace_src_loc_ptr_t src_loc) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_execution_zone_begin_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_execution_zone_begin_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_BEGIN;
  packet->executor_id = trace_buffer->executor_id;
  packet->execution_query_id = execution_query_id;
  packet->issue_timestamp = iree_hal_amdgpu_device_timestamp();
  packet->src_loc = src_loc;
  return iree_hal_amdgpu_device_query_ringbuffer_signal_for_id(
      &trace_buffer->query_ringbuffer, execution_query_id);
}

iree_hsa_signal_t iree_hal_amdgpu_device_trace_execution_zone_end(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_execution_zone_end_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_execution_zone_end_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_END;
  packet->executor_id = trace_buffer->executor_id;
  packet->execution_query_id = execution_query_id;
  packet->issue_timestamp = iree_hal_amdgpu_device_timestamp();
  return iree_hal_amdgpu_device_query_ringbuffer_signal_for_id(
      &trace_buffer->query_ringbuffer, execution_query_id);
}

void iree_hal_amdgpu_device_trace_execution_zone_notify(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id,
    iree_hal_amdgpu_trace_agent_timestamp_t execution_timestamp) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_execution_zone_notify_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_execution_zone_notify_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_NOTIFY;
  packet->executor_id = trace_buffer->executor_id;
  packet->execution_query_id = execution_query_id;
  packet->execution_timestamp = execution_timestamp;
}

IREE_OCL_GLOBAL iree_hal_amdgpu_trace_agent_timestamp_t* IREE_OCL_RESTRICT
iree_hal_amdgpu_device_trace_execution_zone_notify_batch(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id_base,
    uint16_t execution_query_count,
    iree_hal_amdgpu_trace_agent_timestamp_t execution_timestamp) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_execution_zone_notify_batch_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer,
          sizeof(iree_hal_amdgpu_trace_execution_zone_notify_batch_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_NOTIFY;
  packet->executor_id = trace_buffer->executor_id;
  packet->execution_query_id_base = execution_query_id_base;
  packet->execution_query_count = execution_query_count;
  return &packet->execution_timestamps[0];
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_TRACING_FEATURES & \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION

iree_hsa_signal_t iree_hal_amdgpu_device_trace_execution_zone_dispatch(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_execution_zone_type_t zone_type, uint32_t export_loc,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_execution_zone_dispatch_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer,
          sizeof(iree_hal_amdgpu_trace_execution_zone_dispatch_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_DISPATCH;
  packet->zone_type = zone_type;
  packet->executor_id = trace_buffer->executor_id;
  packet->execution_query_id = execution_query_id;
  packet->export_loc = export_loc;
  packet->issue_timestamp = iree_hal_amdgpu_device_timestamp();
  return iree_hal_amdgpu_device_query_ringbuffer_signal_for_id(
      &trace_buffer->query_ringbuffer, execution_query_id);
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_TRACING_FEATURES & \
    IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING

void iree_hal_amdgpu_device_trace_memory_alloc(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal, uint64_t ptr,
    uint64_t size) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_memory_alloc_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_memory_alloc_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_ALLOC;
  packet->pool = name_literal;
  packet->timestamp = iree_hal_amdgpu_device_timestamp();
  packet->ptr = ptr;
  packet->size = size;
}

void iree_hal_amdgpu_device_trace_memory_free(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal, uint64_t ptr) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_memory_free_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_memory_free_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_FREE;
  packet->timestamp = iree_hal_amdgpu_device_timestamp();
  packet->ptr = ptr;
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_TRACING_FEATURES & \
    IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES

void iree_hal_amdgpu_device_trace_message_literal(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_color_t color,
    iree_hal_amdgpu_trace_string_literal_ptr_t value_literal) {
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_message_literal_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_message_literal_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_LITERAL;
  packet->timestamp = iree_hal_amdgpu_device_timestamp();
  packet->value = value_literal;
}

void iree_hal_amdgpu_device_trace_message_dynamic(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_trace_buffer_t* IREE_OCL_RESTRICT
        trace_buffer,
    iree_hal_amdgpu_trace_color_t color,
    IREE_OCL_GLOBAL const char* IREE_OCL_RESTRICT value, size_t value_length) {
  const size_t total_size =
      sizeof(iree_hal_amdgpu_trace_message_dynamic_t) + value_length;
  IREE_OCL_GLOBAL iree_hal_amdgpu_trace_message_dynamic_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(trace_buffer, total_size);
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_DYNAMIC;
  packet->length = (uint32_t)value_length;
  packet->timestamp = iree_hal_amdgpu_device_timestamp();
  iree_ocl_memcpy(packet->value, value, value_length);
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES
