// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_ALLOCATOR_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_ALLOCATOR_H_

#include "iree/hal/drivers/amdgpu/device/buffer.h"
#include "iree/hal/drivers/amdgpu/device/host.h"
#include "iree/hal/drivers/amdgpu/device/support/opencl.h"
#include "iree/hal/drivers/amdgpu/device/support/queue.h"
#include "iree/hal/drivers/amdgpu/device/support/signal.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_allocator_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_device_allocator_pool_s {
  int reserved;
} iree_hal_amdgpu_device_allocator_pool_t;

// DO NOT SUBMIT
typedef struct iree_hal_amdgpu_device_allocator_s {
  // Host that handles pool management operations (grow/trim/etc).
  IREE_OCL_GLOBAL iree_hal_amdgpu_device_host_t* host;

  // DO NOT SUBMIT local pools
} iree_hal_amdgpu_device_allocator_t;

#if defined(IREE_OCL_TARGET_DEVICE)

// Returns true if the operation completed synchronously. If asynchronous the
// provided |scheduler_queue_entry| will be retired after the asynchronous
// operation completes.
bool iree_hal_amdgpu_device_allocator_alloca(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_allocator_t* IREE_OCL_RESTRICT
        allocator,
    uint64_t scheduler_queue_entry, uint32_t pool, uint32_t min_alignment,
    uint64_t allocation_size,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_allocation_handle_t*
        IREE_OCL_RESTRICT out_handle);

// Returns true if the operation completed synchronously. If asynchronous the
// provided |scheduler_queue_entry| will be retired after the asynchronous
// operation completes.
bool iree_hal_amdgpu_device_allocator_dealloca(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_allocator_t* IREE_OCL_RESTRICT
        allocator,
    uint64_t scheduler_queue_entry, uint32_t pool,
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_allocation_handle_t*
        IREE_OCL_RESTRICT handle);

#endif  // IREE_OCL_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_ALLOCATOR_H_
