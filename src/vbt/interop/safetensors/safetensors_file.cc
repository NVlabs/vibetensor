// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/interop/safetensors/file.h"

#include "serialize_prep.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vbt {
namespace interop {
namespace safetensors {

#if VBT_WITH_SAFETENSORS

namespace {

#if defined(__unix__) || defined(__APPLE__)

class Fd {
 public:
  explicit Fd(int fd) : fd_(fd) {}
  ~Fd() {
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
  }

  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;

  Fd(Fd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
  Fd& operator=(Fd&& other) noexcept {
    if (this == &other) return *this;
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
    fd_ = other.fd_;
    other.fd_ = -1;
    return *this;
  }

  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }

 private:
  int fd_ = -1;
};

[[nodiscard]] int open_readonly_no_follow(const char* path) {
  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif

  while (true) {
    const int fd = ::open(path, flags);
    if (fd >= 0) return fd;
    if (errno == EINTR) continue;
    return -1;
  }
}

[[nodiscard]] int open_write_trunc_no_follow(const char* path) {
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif

  while (true) {
    const int fd = ::open(path, flags, 0644);
    if (fd >= 0) return fd;
    if (errno == EINTR) continue;
    return -1;
  }
}

[[nodiscard]] inline bool checked_add_size(std::size_t a, std::size_t b, std::size_t& out) {
  if (a > (std::numeric_limits<std::size_t>::max() - b)) return false;
  out = a + b;
  return true;
}

inline void write_u64_le(std::uint64_t v, std::byte out[8]) {
  for (int i = 0; i < 8; ++i) {
    out[i] = static_cast<std::byte>((v >> (8 * i)) & 0xFF);
  }
}

[[nodiscard]] std::pair<std::shared_ptr<const std::byte>, std::size_t> read_all_bytes(
    const char* path,
    FileOpenOptions opts) {
  if (!path || *path == '\0') {
    throw SafeTensorsError(ErrorCode::IoError, "safetensors: null or empty path");
  }

  const int raw_fd = open_readonly_no_follow(path);
  if (raw_fd < 0) {
    throw SafeTensorsError(ErrorCode::IoError,
                           std::string("safetensors: open failed: ") + std::strerror(errno));
  }
  Fd fd(raw_fd);

  struct stat st {};
  while (true) {
    if (::fstat(fd.get(), &st) == 0) break;
    if (errno == EINTR) continue;
    throw SafeTensorsError(ErrorCode::IoError,
                           std::string("safetensors: fstat failed: ") + std::strerror(errno));
  }

  if (opts.require_regular_file && !S_ISREG(st.st_mode)) {
    throw SafeTensorsError(ErrorCode::NotRegularFile, "safetensors: not a regular file");
  }

  if (st.st_size < 0) {
    throw SafeTensorsError(ErrorCode::IoError, "safetensors: invalid file size");
  }

  const std::uint64_t size_u64 = static_cast<std::uint64_t>(st.st_size);
  if (opts.max_file_size_bytes && size_u64 > static_cast<std::uint64_t>(*opts.max_file_size_bytes)) {
    throw SafeTensorsError(ErrorCode::FileTooLarge, "safetensors: file too large");
  }
  if (size_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw SafeTensorsError(ErrorCode::FileTooLarge, "safetensors: file too large");
  }

  const std::size_t size = static_cast<std::size_t>(size_u64);
  auto buf = std::make_shared<std::vector<std::byte>>(size);

  const std::size_t max_io =
      static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());

  std::size_t pos = 0;
  while (pos < size) {
    const std::size_t want = std::min(size - pos, max_io);
    const ssize_t nread = ::read(fd.get(), buf->data() + pos, want);
    if (nread < 0) {
      if (errno == EINTR) continue;
      throw SafeTensorsError(ErrorCode::IoError,
                             std::string("safetensors: read failed: ") + std::strerror(errno));
    }
    if (nread == 0) {
      throw SafeTensorsError(ErrorCode::IoError, "safetensors: unexpected EOF while reading file");
    }
    pos += static_cast<std::size_t>(nread);
  }

  // Alias the vector's storage as a shared_ptr<const std::byte>.
  std::shared_ptr<const std::byte> owner(buf, buf->data());
  return {std::move(owner), size};
}

[[nodiscard]] std::pair<std::shared_ptr<const std::byte>, std::size_t> mmap_bytes(
    const char* path,
    FileOpenOptions opts) {
#if !VBT_SAFETENSORS_ENABLE_MMAP
  (void)path;
  (void)opts;
  throw SafeTensorsError(ErrorCode::IoError, "safetensors: mmap loading is not available");
#else
  if (!path || *path == '\0') {
    throw SafeTensorsError(ErrorCode::IoError, "safetensors: null or empty path");
  }

  const int raw_fd = open_readonly_no_follow(path);
  if (raw_fd < 0) {
    throw SafeTensorsError(ErrorCode::IoError,
                           std::string("safetensors: open failed: ") + std::strerror(errno));
  }
  Fd fd(raw_fd);

  struct stat st {};
  while (true) {
    if (::fstat(fd.get(), &st) == 0) break;
    if (errno == EINTR) continue;
    throw SafeTensorsError(ErrorCode::IoError,
                           std::string("safetensors: fstat failed: ") + std::strerror(errno));
  }

  if (opts.require_regular_file && !S_ISREG(st.st_mode)) {
    throw SafeTensorsError(ErrorCode::NotRegularFile, "safetensors: not a regular file");
  }

  if (st.st_size < 0) {
    throw SafeTensorsError(ErrorCode::IoError, "safetensors: invalid file size");
  }

  const std::uint64_t size_u64 = static_cast<std::uint64_t>(st.st_size);
  if (opts.max_file_size_bytes &&
      size_u64 > static_cast<std::uint64_t>(*opts.max_file_size_bytes)) {
    throw SafeTensorsError(ErrorCode::FileTooLarge, "safetensors: file too large");
  }
  if (size_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw SafeTensorsError(ErrorCode::FileTooLarge, "safetensors: file too large");
  }

  const std::size_t size = static_cast<std::size_t>(size_u64);
  if (size == 0) {
    auto buf = std::make_shared<std::vector<std::byte>>();
    std::shared_ptr<const std::byte> owner(buf, buf->data());
    return {std::move(owner), 0};
  }

  void* addr = MAP_FAILED;
  while (addr == MAP_FAILED) {
    addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd.get(), 0);
    if (addr != MAP_FAILED) break;
    if (errno == EINTR) continue;
    throw SafeTensorsError(ErrorCode::IoError,
                           std::string("safetensors: mmap failed: ") + std::strerror(errno));
  }
  if (addr == nullptr) {
    (void)::munmap(addr, size);
    throw SafeTensorsError(ErrorCode::IoError, "safetensors: mmap returned null address");
  }

  const std::byte* p = reinterpret_cast<const std::byte*>(addr);
  std::shared_ptr<const std::byte> owner(p, [size](const std::byte* ptr) noexcept {
    (void)::munmap(const_cast<void*>(reinterpret_cast<const void*>(ptr)), size);
  });
  return {std::move(owner), size};
#endif
}

void write_all_bytes(int fd,
                     std::span<const std::byte> data,
                     std::size_t chunk_bytes,
                     const char* what) {
  const std::size_t max_io =
      static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
  const std::size_t chunk = std::max<std::size_t>(std::min(chunk_bytes, max_io), 1);

  std::size_t pos = 0;
  while (pos < data.size()) {
    const std::size_t remaining = data.size() - pos;
    const std::size_t to_write = std::min(chunk, remaining);

    const ssize_t nw = ::write(fd, data.data() + pos, to_write);
    if (nw < 0) {
      if (errno == EINTR) continue;
      throw SafeTensorsError(ErrorCode::IoError,
                             std::string("safetensors: ") + what + ": " + std::strerror(errno));
    }
    if (nw == 0) {
      throw SafeTensorsError(ErrorCode::IoError,
                             std::string("safetensors: ") + what + ": wrote 0 bytes");
    }

    pos += static_cast<std::size_t>(nw);
  }
}

#else

[[nodiscard]] std::pair<std::shared_ptr<const std::byte>, std::size_t> read_all_bytes(
    const char* /*path*/,
    FileOpenOptions /*opts*/) {
  throw SafeTensorsError(ErrorCode::IoError,
                         "safetensors: file I/O is not supported on this platform");
}

[[nodiscard]] std::pair<std::shared_ptr<const std::byte>, std::size_t> mmap_bytes(
    const char* /*path*/,
    FileOpenOptions /*opts*/) {
  throw SafeTensorsError(ErrorCode::IoError,
                         "safetensors: file I/O is not supported on this platform");
}

#endif

} // namespace

SafeTensorsFile SafeTensorsFile::open(const char* path,
                                      FileLoadMode mode,
                                      ParseOptions parse_opts,
                                      FileOpenOptions open_opts) {
  std::shared_ptr<const std::byte> owner;
  std::size_t size = 0;

  if (mode == FileLoadMode::ReadAll) {
    auto [o, s] = read_all_bytes(path, open_opts);
    owner = std::move(o);
    size = s;
  } else if (mode == FileLoadMode::MMap) {
    auto [o, s] = mmap_bytes(path, open_opts);
    owner = std::move(o);
    size = s;
  } else if (mode == FileLoadMode::MMapIfAvailable) {
    try {
      auto [o, s] = mmap_bytes(path, open_opts);
      owner = std::move(o);
      size = s;
    } catch (const SafeTensorsError& e) {
      if (e.code() != ErrorCode::IoError) throw;
      auto [o, s] = read_all_bytes(path, open_opts);
      owner = std::move(o);
      size = s;
    }
  } else {
    throw SafeTensorsError(ErrorCode::IoError, "safetensors: unknown FileLoadMode");
  }

  SafeTensorsFile out;
  out.owner_ = std::move(owner);
  out.size_ = size;
  out.view_ = SafeTensorsView::deserialize(std::span<const std::byte>(out.owner_.get(), out.size_),
                                          parse_opts);
  return out;
}

void serialize_to_file(
    const char* path,
    std::span<const TensorEntry> tensors,
    const std::optional<std::vector<std::pair<std::string, std::string>>>& user_metadata,
    SerializeOptions sopts,
    FileWriteOptions fopts) {
  if (!path || *path == '\0') {
    throw SafeTensorsError(ErrorCode::IoError, "safetensors: null or empty path");
  }

#if defined(__unix__) || defined(__APPLE__)
  detail::PreparedSerialization prep = detail::prepare_serialization(tensors, user_metadata, sopts);

  // Pre-size checks (overflow safety). This matches the in-memory `serialize(...)`
  // output size for the same inputs.
  std::size_t total_size = 0;
  std::size_t tmp = 0;
  if (!checked_add_size(static_cast<std::size_t>(8), prep.header_bytes.size(), tmp) ||
      !checked_add_size(tmp, prep.actual_data_bytes, total_size)) {
    throw SafeTensorsError(ErrorCode::ValidationOverflow, "safetensors: output size overflow");
  }

  const int raw_fd = open_write_trunc_no_follow(path);
  if (raw_fd < 0) {
    throw SafeTensorsError(ErrorCode::IoError,
                           std::string("safetensors: open for write failed: ") +
                               std::strerror(errno));
  }
  Fd fd(raw_fd);

  if (fopts.preallocate) {
    const std::uint64_t max_off =
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
    if (static_cast<std::uint64_t>(total_size) > max_off) {
      throw SafeTensorsError(ErrorCode::FileTooLarge, "safetensors: file too large");
    }
    while (true) {
      if (::ftruncate(fd.get(), static_cast<off_t>(total_size)) == 0) break;
      if (errno == EINTR) continue;
      throw SafeTensorsError(ErrorCode::IoError,
                             std::string("safetensors: ftruncate failed: ") + std::strerror(errno));
    }
  }

  // Write framing: 8-byte little-endian header length + header bytes + tensor data.
  std::byte header_len_le[8];
  write_u64_le(static_cast<std::uint64_t>(prep.header_bytes.size()), header_len_le);
  write_all_bytes(fd.get(), std::span<const std::byte>(header_len_le, 8), fopts.buf_bytes,
                  "write failed");
  write_all_bytes(fd.get(),
                  std::span<const std::byte>(prep.header_bytes.data(), prep.header_bytes.size()),
                  fopts.buf_bytes, "write failed");
  for (const auto& pt : prep.tensors) {
    write_all_bytes(fd.get(), pt.entry->data, fopts.buf_bytes, "write failed");
  }
#else
  (void)tensors;
  (void)user_metadata;
  (void)sopts;
  (void)fopts;
  throw SafeTensorsError(ErrorCode::IoError, "safetensors: file I/O is not supported on this platform");
#endif
}

#else

SafeTensorsFile SafeTensorsFile::open(const char* /*path*/,
                                      FileLoadMode /*mode*/,
                                      ParseOptions /*parse_opts*/,
                                      FileOpenOptions /*open_opts*/) {
  throw SafeTensorsError(ErrorCode::IoError, "safetensors: support disabled at build time");
}

void serialize_to_file(
    const char* /*path*/,
    std::span<const TensorEntry> /*tensors*/,
    const std::optional<std::vector<std::pair<std::string, std::string>>>& /*user_metadata*/,
    SerializeOptions /*sopts*/,
    FileWriteOptions /*fopts*/) {
  throw SafeTensorsError(ErrorCode::IoError, "safetensors: support disabled at build time");
}

#endif

} // namespace safetensors
} // namespace interop
} // namespace vbt
