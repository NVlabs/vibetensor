// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "include/cuda_allocator_perf.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace vbt {
namespace cuda {
namespace testonly {

namespace {

using vbt::cuda::device_count;

std::string classify_backend_kind_from_env() {
#if !VBT_WITH_CUDA
  (void)device_count;
  return "none";
#else
  if (device_count() <= 0) {
    return "none";
  }
  const char* conf = std::getenv("VBT_CUDA_ALLOC_CONF");
  if (!conf || *conf == '\0') {
    return "native";
  }
  std::string s(conf);
  for (char& c : s) {
    if (c == ',') c = ' ';
  }
  std::istringstream iss(s);
  std::string tok;
  std::string backend_token;
  while (iss >> tok) {
    std::size_t eq = tok.find('=');
    if (eq == std::string::npos) continue;
    std::string key = tok.substr(0, eq);
    std::string val = tok.substr(eq + 1);
    if (key == "backend") {
      backend_token = val;
      break;
    }
  }
  if (backend_token.empty()) {
    return "native";
  }
  for (char& c : backend_token) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (backend_token == "native" || backend_token == "cudamalloc") {
    return "native";
  }
  if (backend_token == "cudamallocasync") {
    return "async";
  }
  return "unknown";
#endif
}

struct ParsedArgs {
  std::string scenario;   // should be "B4" for this binary
  int         device{0};
  std::string run_mode{"normal"};
  int         warmup_override{0};
  int         measure_override{0};
  int         repeats_override{0};
  int         num_replays{1};
  std::string json_out{"-"};
  std::string notes;
  bool        show_help{false};
};

void print_help(const char* prog) {
  std::cout << "Usage: " << prog
            << " --scenario B4 --device <idx> [options]\n"
            << "Options:\n"
            << "  --run-mode smoke|normal|heavy   (default: normal)\n"
            << "  --warmup-iters N               (override defaults; ignored in smoke)\n"
            << "  --measure-iters N              (override defaults; ignored in smoke)\n"
            << "  --repeats N                    (override defaults; ignored in smoke)\n"
            << "  --num-replays N                (graphs scenarios; default 1)\n"
            << "  --json-out PATH|-              (default: - for stdout)\n"
            << "  --notes TEXT                   (optional note string)\n"
            << "  --help                         (show this message)\n";
}

bool parse_int(const char* arg, int& out) {
  if (!arg) return false;
  char* end = nullptr;
  long v = std::strtol(arg, &end, 10);
  if (end == arg || *end != '\0') return false;
  out = static_cast<int>(v);
  return true;
}

ParsedArgs parse_args(int argc, char** argv) {
  ParsedArgs pa;

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strcmp(a, "--help") == 0) {
      pa.show_help = true;
      continue;
    }
    auto need_val = [&](int& idx) -> const char* {
      if (idx + 1 >= argc) return nullptr;
      return argv[++idx];
    };

    if (std::strcmp(a, "--scenario") == 0) {
      const char* v = need_val(i);
      if (!v) throw std::runtime_error("--scenario requires a value");
      pa.scenario = v;
    } else if (std::strcmp(a, "--device") == 0) {
      const char* v = need_val(i);
      if (!v || !parse_int(v, pa.device)) {
        throw std::runtime_error("--device requires an integer value");
      }
    } else if (std::strcmp(a, "--run-mode") == 0) {
      const char* v = need_val(i);
      if (!v) throw std::runtime_error("--run-mode requires a value");
      pa.run_mode = v;
    } else if (std::strcmp(a, "--warmup-iters") == 0) {
      const char* v = need_val(i);
      if (!v || !parse_int(v, pa.warmup_override)) {
        throw std::runtime_error("--warmup-iters requires an integer value");
      }
    } else if (std::strcmp(a, "--measure-iters") == 0) {
      const char* v = need_val(i);
      if (!v || !parse_int(v, pa.measure_override)) {
        throw std::runtime_error("--measure-iters requires an integer value");
      }
    } else if (std::strcmp(a, "--repeats") == 0) {
      const char* v = need_val(i);
      if (!v || !parse_int(v, pa.repeats_override)) {
        throw std::runtime_error("--repeats requires an integer value");
      }
    } else if (std::strcmp(a, "--num-replays") == 0) {
      const char* v = need_val(i);
      if (!v || !parse_int(v, pa.num_replays)) {
        throw std::runtime_error("--num-replays requires an integer value");
      }
    } else if (std::strcmp(a, "--json-out") == 0) {
      const char* v = need_val(i);
      if (!v) throw std::runtime_error("--json-out requires a value");
      pa.json_out = v;
    } else if (std::strcmp(a, "--notes") == 0) {
      const char* v = need_val(i);
      if (!v) throw std::runtime_error("--notes requires a value");
      pa.notes = v;
    } else {
      std::ostringstream oss;
      oss << "unknown argument: " << a;
      throw std::runtime_error(oss.str());
    }
  }

  return pa;
}

ScenarioId parse_scenario_id(const std::string& s) {
  if (s == "B4") return ScenarioId::B4;
  throw std::runtime_error("async perf binary only supports scenario B4");
}

RunMode parse_run_mode(const std::string& s) {
  if (s == "smoke") return RunMode::Smoke;
  if (s == "normal") return RunMode::Normal;
  if (s == "heavy") return RunMode::Heavy;
  throw std::runtime_error("unknown run-mode: " + s);
}

int run_main_impl(int argc, char** argv) {
  ParsedArgs args;
  try {
    args = parse_args(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 2;  // config/CLI error
  }

  if (args.show_help) {
    print_help(argv[0]);
    return 0;
  }

  if (args.scenario.empty()) {
    std::cerr << "--scenario is required\n";
    return 2;
  }

  ScenarioId sid;
  try {
    sid = parse_scenario_id(args.scenario);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 2;
  }

#if !VBT_WITH_CUDA
  (void)args;
  (void)sid;
  std::cerr << "CUDA is not enabled in this build; skipping perf run\n";
  return 77;
#else
  if (device_count() <= 0) {
    std::cerr << "No CUDA devices detected; skipping perf run\n";
    return 77;
  }
  if (args.device < 0 || args.device >= device_count()) {
    std::cerr << "Requested device index is out of range; skipping perf run\n";
    return 77;
  }

  std::string backend_kind = classify_backend_kind_from_env();
  if (backend_kind != "async") {
    std::cerr << "async perf binary requires backend=cudaMallocAsync; skipping perf run\n";
    return 77;
  }

  RunMode mode;
  try {
    mode = parse_run_mode(args.run_mode);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 2;
  }

  RunCounts overrides;
  overrides.warmup_iters = args.warmup_override;
  overrides.measure_iters = args.measure_override;
  overrides.repeats = args.repeats_override;

  RunCounts counts;
  try {
    counts = resolve_run_counts(sid, mode, overrides);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 2;  // guardrail violation
  }

  PerfConfig cfg;
  cfg.scenario_id = sid;
  cfg.runner = Runner::CppAsync;
  cfg.run_mode = mode;
  cfg.device_index = args.device;
  cfg.counts = counts;
  cfg.num_replays = args.num_replays;
  cfg.notes = args.notes;

  PerfResult result;
  try {
    result = run_B4(cfg);
  } catch (const std::exception& ex) {
    std::cerr << "runtime error: " << ex.what() << "\n";
    return 1;
  }

  if (args.json_out == "-" || args.json_out.empty()) {
    write_perf_result_json(result, std::cout);
    if (!std::cout.good()) {
      std::cerr << "json-out: write failed to stdout\n";
      return 1;
    }
  } else {
    std::ofstream ofs(args.json_out, std::ios::out | std::ios::trunc);
    if (!ofs.good()) {
      std::cerr << "json-out: write failed (unable to open path)\n";
      return 1;
    }
    write_perf_result_json(result, ofs);
    if (!ofs.good()) {
      std::cerr << "json-out: write failed during write()\n";
      return 1;
    }
  }

  return 0;
#endif
}

} // anonymous namespace

// Public entry point used by main().
int run_main(int argc, char** argv) {
  return run_main_impl(argc, argv);
}

} // namespace testonly
} // namespace cuda
} // namespace vbt

int main(int argc, char** argv) {
  return vbt::cuda::testonly::run_main(argc, argv);
}
