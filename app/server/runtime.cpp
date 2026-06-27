#include "runtime.h"

#include "../cli/request.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace minitts::server {
namespace {

using engine::io::json::Value;

std::string json_quote(std::string_view value) {
    return engine::io::json::stringify_string(value);
}

std::filesystem::path resolve_path(const std::filesystem::path & base, const std::filesystem::path & path) {
    return path.is_absolute() ? path : base / path;
}

std::unordered_map<std::string, std::string> options_from_object(const Value * value) {
    return minitts::cli::json_options_map(value);
}

void add_option_from_json(
    std::unordered_map<std::string, std::string> & options,
    const Value & object,
    const std::string & field,
    const std::string & option_key) {
    const auto * value = object.find(field);
    if (value != nullptr && !value->is_null()) {
        options[option_key] = minitts::cli::json_option_string(*value);
    }
}

engine::core::BackendType parse_backend_type(std::string backend) {
    std::transform(backend.begin(), backend.end(), backend.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (backend == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (backend == "cuda" || backend == "gpu") {
        return engine::core::BackendType::Cuda;
    }
    if (backend == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (backend == "metal") {
        return engine::core::BackendType::Metal;
    }
    if (backend == "best" || backend == "auto") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("unsupported server backend: " + backend);
}

std::vector<uint8_t> encode_pcm16_wav(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("audio output sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("audio output channel count must be positive");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("audio output sample count must be divisible by channel count");
    }

    const uint16_t channels = static_cast<uint16_t>(audio.channels);
    const uint16_t bits_per_sample = 16;
    const uint32_t data_bytes = static_cast<uint32_t>(audio.samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_bytes;
    const uint32_t byte_rate = static_cast<uint32_t>(audio.sample_rate) * channels * bits_per_sample / 8;
    const uint16_t block_align = channels * bits_per_sample / 8;

    std::vector<uint8_t> out;
    out.reserve(44 + data_bytes);
    auto append_bytes = [&](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    auto append_u16 = [&](uint16_t value) { append_bytes(&value, sizeof(value)); };
    auto append_u32 = [&](uint32_t value) { append_bytes(&value, sizeof(value)); };

    out.insert(out.end(), {'R', 'I', 'F', 'F'});
    append_u32(riff_size);
    out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    append_u32(16);
    append_u16(1);
    append_u16(channels);
    append_u32(static_cast<uint32_t>(audio.sample_rate));
    append_u32(byte_rate);
    append_u16(block_align);
    append_u16(bits_per_sample);
    out.insert(out.end(), {'d', 'a', 't', 'a'});
    append_u32(data_bytes);
    for (float sample : audio.samples) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
        append_bytes(&pcm, sizeof(pcm));
    }
    return out;
}

std::string encode_pcm16_payload(const engine::runtime::AudioBuffer & audio) {
    if (audio.channels <= 0) {
        throw std::runtime_error("audio output channel count must be positive");
    }
    std::string out;
    out.reserve(audio.samples.size() * sizeof(int16_t));
    for (float sample : audio.samples) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
        out.append(reinterpret_cast<const char *>(&pcm), sizeof(pcm));
    }
    return out;
}

struct LeadingSilenceFilterConfig {
    bool enabled = true;
    float threshold = 80.0F / 32767.0F;
    double preroll_ms = 20.0;
    double scan_window_ms = 5.0;
    double min_trim_ms = 10.0;
    double fade_ms = 25.0;
};

class LeadingSilenceFilter {
public:
    explicit LeadingSilenceFilter(LeadingSilenceFilterConfig config) : config_(config) {}

    std::optional<engine::runtime::AudioBuffer> process(const engine::runtime::AudioBuffer & audio) {
        if (!config_.enabled || started_) {
            return audio;
        }
        if (audio.sample_rate <= 0) {
            throw std::runtime_error("stream audio sample rate must be positive");
        }
        if (audio.channels <= 0) {
            throw std::runtime_error("stream audio channel count must be positive");
        }
        if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
            throw std::runtime_error("stream audio sample count must be divisible by channel count");
        }
        if (audio.samples.empty()) {
            return std::nullopt;
        }

        const int64_t frames = static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
        if (average_abs(audio.samples, 0, frames, audio.channels) < config_.threshold) {
            dropped_frames_ += frames;
            return std::nullopt;
        }

        engine::runtime::AudioBuffer output = audio;
        const int64_t trim_frames = find_trim_frames(output.samples, frames, output.channels, output.sample_rate);
        if (trim_frames > 0) {
            output.samples.erase(
                output.samples.begin(),
                output.samples.begin() + static_cast<std::ptrdiff_t>(trim_frames * output.channels));
            dropped_frames_ += trim_frames;
        }
        if (dropped_frames_ > 0) {
            apply_fade_in(output);
        }
        started_ = true;
        return output;
    }

    double dropped_ms(int sample_rate) const {
        if (sample_rate <= 0) {
            return 0.0;
        }
        return static_cast<double>(dropped_frames_) * 1000.0 / static_cast<double>(sample_rate);
    }

private:
    static float average_abs(
        const std::vector<float> & samples,
        int64_t start_frame,
        int64_t frame_count,
        int channels) {
        if (frame_count <= 0 || channels <= 0) {
            return 0.0F;
        }
        double sum = 0.0;
        const int64_t start = start_frame * channels;
        const int64_t end = start + frame_count * channels;
        for (int64_t index = start; index < end; ++index) {
            sum += std::abs(static_cast<double>(samples[static_cast<size_t>(index)]));
        }
        return static_cast<float>(sum / static_cast<double>(frame_count * channels));
    }

    int64_t find_trim_frames(
        const std::vector<float> & samples,
        int64_t frames,
        int channels,
        int sample_rate) const {
        const int64_t window_frames = std::max<int64_t>(
            1,
            static_cast<int64_t>(std::llround(static_cast<double>(sample_rate) * config_.scan_window_ms / 1000.0)));
        const int64_t preroll_frames = std::max<int64_t>(
            0,
            static_cast<int64_t>(std::llround(static_cast<double>(sample_rate) * config_.preroll_ms / 1000.0)));

        for (int64_t start = 0; start < frames; start += window_frames) {
            const int64_t count = std::min<int64_t>(window_frames, frames - start);
            if (average_abs(samples, start, count, channels) >= config_.threshold) {
                const int64_t trim = std::max<int64_t>(0, start - preroll_frames);
                const double trim_ms = static_cast<double>(trim) * 1000.0 / static_cast<double>(sample_rate);
                return trim_ms >= config_.min_trim_ms ? trim : 0;
            }
        }
        return 0;
    }

    void apply_fade_in(engine::runtime::AudioBuffer & audio) const {
        if (audio.samples.empty() || audio.sample_rate <= 0 || audio.channels <= 0) {
            return;
        }
        const int64_t frames = static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
        const int64_t fade_frames = std::min<int64_t>(
            frames,
            std::max<int64_t>(
                1,
                static_cast<int64_t>(std::llround(static_cast<double>(audio.sample_rate) * config_.fade_ms / 1000.0))));
        for (int64_t frame = 0; frame < fade_frames; ++frame) {
            const float gain = static_cast<float>(frame + 1) / static_cast<float>(fade_frames);
            for (int channel = 0; channel < audio.channels; ++channel) {
                audio.samples[static_cast<size_t>(frame * audio.channels + channel)] *= gain;
            }
        }
    }

    LeadingSilenceFilterConfig config_;
    bool started_ = false;
    int64_t dropped_frames_ = 0;
};

LeadingSilenceFilterConfig parse_leading_silence_filter_config(const Value & body) {
    LeadingSilenceFilterConfig config;
    config.enabled = engine::io::json::optional_bool(body, "trim_leading_silence", true);
    config.threshold = engine::io::json::optional_f32(body, "leading_silence_threshold", config.threshold);
    config.preroll_ms = engine::io::json::optional_f32(body, "leading_silence_preroll_ms", static_cast<float>(config.preroll_ms));
    config.scan_window_ms = engine::io::json::optional_f32(
        body,
        "leading_silence_scan_window_ms",
        static_cast<float>(config.scan_window_ms));
    config.min_trim_ms = engine::io::json::optional_f32(
        body,
        "leading_silence_min_trim_ms",
        static_cast<float>(config.min_trim_ms));
    config.fade_ms = engine::io::json::optional_f32(body, "leading_silence_fade_ms", static_cast<float>(config.fade_ms));
    return config;
}

std::string base64_encode(const uint8_t * data, size_t size) {
    constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((size + 2) / 3) * 4);
    for (size_t i = 0; i < size; i += 3) {
        const uint32_t b0 = data[i];
        const uint32_t b1 = i + 1 < size ? data[i + 1] : 0;
        const uint32_t b2 = i + 2 < size ? data[i + 2] : 0;
        const uint32_t chunk = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kAlphabet[(chunk >> 18) & 0x3f]);
        out.push_back(kAlphabet[(chunk >> 12) & 0x3f]);
        out.push_back(i + 1 < size ? kAlphabet[(chunk >> 6) & 0x3f] : '=');
        out.push_back(i + 2 < size ? kAlphabet[chunk & 0x3f] : '=');
    }
    return out;
}

std::string base64_encode(const std::vector<uint8_t> & bytes) {
    return base64_encode(bytes.data(), bytes.size());
}

std::string task_result_json(const engine::runtime::TaskResult & result) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    auto field = [&](const std::string & name) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << json_quote(name) << ":";
    };

    if (result.text_output.has_value()) {
        field("text");
        out << json_quote(result.text_output->text);
        if (!result.text_output->language.empty()) {
            field("language");
            out << json_quote(result.text_output->language);
        }
    }
    if (result.audio_output.has_value()) {
        const auto wav = encode_pcm16_wav(*result.audio_output);
        field("audio");
        out << json_quote(base64_encode(wav));
        field("sample_rate");
        out << result.audio_output->sample_rate;
        field("channels");
        out << result.audio_output->channels;
    }
    if (!result.named_audio_outputs.empty()) {
        field("named_audio_outputs");
        out << "[";
        for (size_t i = 0; i < result.named_audio_outputs.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto wav = encode_pcm16_wav(result.named_audio_outputs[i].audio);
            out << "{\"id\":" << json_quote(result.named_audio_outputs[i].id)
                << ",\"audio\":" << json_quote(base64_encode(wav))
                << ",\"sample_rate\":" << result.named_audio_outputs[i].audio.sample_rate
                << ",\"channels\":" << result.named_audio_outputs[i].audio.channels
                << "}";
        }
        out << "]";
    }
    if (!result.speech_segments.empty()) {
        field("segments");
        out << "[";
        for (size_t i = 0; i < result.speech_segments.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & segment = result.speech_segments[i];
            out << "{\"start_sample\":" << segment.span.start_sample
                << ",\"end_sample\":" << segment.span.end_sample
                << ",\"confidence\":" << segment.confidence << "}";
        }
        out << "]";
    }
    if (!result.speaker_turns.empty()) {
        field("speaker_turns");
        out << "[";
        for (size_t i = 0; i < result.speaker_turns.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & turn = result.speaker_turns[i];
            out << "{\"start_sample\":" << turn.span.start_sample
                << ",\"end_sample\":" << turn.span.end_sample
                << ",\"speaker_id\":" << json_quote(turn.speaker_id)
                << ",\"confidence\":" << turn.confidence << "}";
        }
        out << "]";
    }
    if (!result.word_timestamps.empty()) {
        field("words");
        out << "[";
        for (size_t i = 0; i < result.word_timestamps.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & word = result.word_timestamps[i];
            out << "{\"word\":" << json_quote(word.word)
                << ",\"start_sample\":" << word.span.start_sample
                << ",\"end_sample\":" << word.span.end_sample
                << ",\"confidence\":" << word.confidence << "}";
        }
        out << "]";
    }
    out << "}";
    return out.str();
}

const engine::runtime::AudioBuffer & select_audio_output(const engine::runtime::TaskResult & result) {
    if (result.audio_output.has_value()) {
        return *result.audio_output;
    }
    if (result.named_audio_outputs.size() == 1) {
        return result.named_audio_outputs.front().audio;
    }
    throw std::runtime_error("model result did not contain exactly one audio output");
}

engine::runtime::TaskRequest build_openai_speech_request(const Value & body, const std::filesystem::path & base_dir) {
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{
        engine::io::json::require_string(body, "input"),
        engine::io::json::optional_string(body, "language", ""),
    };

    engine::runtime::VoiceCondition voice;
    bool has_voice = false;
    std::string voice_ref_path_option;
    if (const auto * value = body.find("voice")) {
        engine::runtime::VoiceReference reference;
        reference.cached_voice_id = value->as_string();
        voice.speaker = std::move(reference);
        has_voice = true;
    }
    if (const auto * value = body.find("voice_ref_path")) {
        voice_ref_path_option = resolve_path(base_dir, value->as_string()).string();
    } else if (const auto * value = body.find("voice_ref")) {
        if (!voice.speaker.has_value()) {
            voice.speaker = engine::runtime::VoiceReference{};
        }
        voice.speaker->audio = minitts::cli::read_audio_buffer(resolve_path(base_dir, value->as_string()));
        has_voice = true;
    }
    if (has_voice) {
        request.voice = std::move(voice);
    }

    request.options = options_from_object(body.find("options"));
    add_option_from_json(request.options, body, "seed", "seed");
    add_option_from_json(request.options, body, "temperature", "temperature");
    add_option_from_json(request.options, body, "top_k", "top_k");
    add_option_from_json(request.options, body, "top_p", "top_p");
    add_option_from_json(request.options, body, "max_tokens", "max_tokens");
    add_option_from_json(request.options, body, "max_steps", "max_steps");
    add_option_from_json(request.options, body, "repetition_penalty", "repetition_penalty");
    add_option_from_json(request.options, body, "guidance_scale", "guidance_scale");
    add_option_from_json(request.options, body, "num_inference_steps", "num_inference_steps");
    if (!voice_ref_path_option.empty()) {
        request.options["voice_ref_path"] = voice_ref_path_option;
    }
    if (const auto * value = body.find("instructions")) {
        request.options["instruct"] = value->as_string();
    }
    if (const auto * value = body.find("reference_text")) {
        request.options["reference_text"] = value->as_string();
    }
    return request;
}

engine::runtime::TaskRequest build_openai_transcription_request(const Value & body, const std::filesystem::path & base_dir) {
    const auto * audio = body.find("audio");
    if (audio == nullptr) {
        audio = body.find("audio_path");
    }
    if (audio == nullptr) {
        audio = body.find("file");
    }
    if (audio == nullptr || !audio->is_string()) {
        throw std::runtime_error("transcription request requires audio, audio_path, or file path");
    }

    engine::runtime::TaskRequest request;
    request.audio_input = minitts::cli::read_audio_buffer(resolve_path(base_dir, audio->as_string()));
    request.options = options_from_object(body.find("options"));
    if (const auto * value = body.find("language")) {
        request.options["language"] = value->as_string();
    }
    return request;
}

}  // namespace

ServerState::ServerState(ServerConfig config, std::filesystem::path request_base)
    : config_(std::move(config)),
      request_base_(std::move(request_base)) {
    load_models();
}

bool ServerState::handle_stream(const HttpRequest & request, HttpResponder & responder) {
    if (request.method == "POST" && request.path == "/v1/audio/speech/stream") {
        handle_speech_stream(request.body, responder);
        return true;
    }
    return false;
}

HttpResponse ServerState::handle(const HttpRequest & request) {
    if (request.method == "GET" && request.path == "/health") {
        return json_response(
            "{\"status\":\"ok\",\"backend\":" + json_quote(config_.backend) +
            ",\"models\":" + std::to_string(models_.size()) + "}");
    }
    if (request.method == "GET" && request.path == "/v1/models") {
        return json_response(models_json());
    }
    if (request.method == "POST" && request.path == "/v1/audio/speech") {
        return handle_speech(request.body);
    }
    if (request.method == "POST" && request.path == "/v1/audio/transcriptions") {
        return handle_transcription(request.body);
    }
    if (request.method == "POST" && request.path == "/v1/tasks/run") {
        return handle_generic_run(request.body);
    }
    return error_response(404, "unknown endpoint: " + request.path, "not_found");
}

void ServerState::load_models() {
    auto registry = engine::runtime::make_default_registry();
    for (auto & config : config_.models) {
        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = config.path;
        load_request.family_hint = config.family;
        load_request.config_id = config.config_id;
        load_request.weight_id = config.weight_id;
        load_request.options = config.load_options;

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = parse_backend_type(config_.backend);
        session_options.backend.device = config_.device;
        session_options.backend.threads = config_.threads;
        session_options.options = config.session_options;

        auto loaded = std::make_unique<LoadedModel>();
        loaded->config = std::move(config);
        loaded->task = engine::runtime::TaskSpec{
            engine::runtime::parse_voice_task_kind(loaded->config.task),
            engine::runtime::parse_run_mode(loaded->config.mode),
        };
        loaded->model = registry.load(load_request);
        loaded->session = loaded->model->create_task_session(loaded->task, session_options);
        loaded->offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(loaded->session.get());
        loaded->streaming_output =
            dynamic_cast<engine::runtime::IStreamingOutputVoiceTaskSession *>(loaded->session.get());
        if (loaded->offline == nullptr && loaded->streaming_output == nullptr) {
            throw std::runtime_error("configured model provides no server-compatible execution path: " + loaded->config.id);
        }
        if (!model_index_.emplace(loaded->config.id, models_.size()).second) {
            throw std::runtime_error("duplicate server model id: " + loaded->config.id);
        }
        models_.push_back(std::move(loaded));
    }
}

ServerState::LoadedModel & ServerState::require_model(const Value & body) {
    const std::string id = engine::io::json::require_string(body, "model");
    const auto it = model_index_.find(id);
    if (it == model_index_.end()) {
        throw std::runtime_error("unknown model id: " + id);
    }
    return *models_.at(it->second);
}

engine::runtime::TaskResult ServerState::run_model(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request) {
    if (model.offline == nullptr) {
        throw std::runtime_error("configured model does not provide offline execution: " + model.config.id);
    }
    std::lock_guard<std::mutex> lock(model.mutex);
    model.session->prepare(engine::runtime::build_preparation_request(request));
    return model.offline->run(request);
}

HttpResponse ServerState::handle_speech(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    auto & model = require_model(body);
    const auto request = build_openai_speech_request(body, request_base_);
    const auto result = run_model(model, request);
    const auto wav = encode_pcm16_wav(select_audio_output(result));
    const auto response_format = engine::io::json::optional_string(body, "response_format", "wav");
    if (response_format == "json" || response_format == "b64_json") {
        return json_response("{\"audio\":" + json_quote(base64_encode(wav)) + ",\"format\":\"wav\"}");
    }
    return HttpResponse{200, "audio/wav", std::string(reinterpret_cast<const char *>(wav.data()), wav.size()), {}};
}

void ServerState::handle_speech_stream(const std::string & body_text, HttpResponder & responder) {
    const auto body = engine::io::json::parse(body_text);
    auto & model = require_model(body);
    if (model.streaming_output == nullptr) {
        const auto response = error_response(
            400,
            "configured model does not provide streaming output: " + model.config.id,
            "unsupported_model");
        responder.send_response(response.status, response.content_type, response.body, response.headers);
        return;
    }

    const auto request = build_openai_speech_request(body, request_base_);
    auto silence_filter = LeadingSilenceFilter(parse_leading_silence_filter_config(body));
    bool started = false;
    std::lock_guard<std::mutex> lock(model.mutex);
    model.session->prepare(engine::runtime::build_preparation_request(request));
    (void) model.streaming_output->run_streaming_output(
        request,
        [&](const engine::runtime::StreamEvent & event) {
            if (!event.audio_output.has_value()) {
                return true;
            }
            auto filtered = silence_filter.process(*event.audio_output);
            if (!filtered.has_value() || filtered->samples.empty()) {
                return true;
            }
            const auto & audio = *filtered;
            const auto chunk = encode_pcm16_payload(audio);
            if (chunk.empty()) {
                return true;
            }
            if (!started) {
                responder.start_chunked(
                    200,
                    "application/octet-stream",
                    {
                        {"X-Audio-Format", "pcm_s16le"},
                        {"X-Audio-Sample-Rate", std::to_string(audio.sample_rate)},
                        {"X-Audio-Channels", std::to_string(audio.channels)},
                        {"X-Audio-Leading-Silence-Dropped-Ms", std::to_string(silence_filter.dropped_ms(audio.sample_rate))},
                    });
                started = true;
            }
            responder.send_chunk(chunk);
            return true;
        });
    if (started) {
        responder.finish_chunked();
    } else {
        responder.send_response(204, "application/octet-stream", "");
    }
}

HttpResponse ServerState::handle_transcription(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    auto & model = require_model(body);
    const auto request = build_openai_transcription_request(body, request_base_);
    const auto result = run_model(model, request);
    if (!result.text_output.has_value()) {
        throw std::runtime_error("model result did not contain transcript text");
    }
    return json_response("{\"text\":" + json_quote(result.text_output->text) + "}");
}

HttpResponse ServerState::handle_generic_run(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    auto & model = require_model(body);
    const auto * request_json = body.find("request");
    const auto request = minitts::cli::build_request_from_json(
        request_json != nullptr ? *request_json : body,
        request_base_);
    return json_response(task_result_json(run_model(model, request)));
}

std::string ServerState::models_json() const {
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < models_.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        const auto & model = *models_[i];
        out << "{\"id\":" << json_quote(model.config.id)
            << ",\"object\":\"model\""
            << ",\"owned_by\":\"engine\""
            << ",\"family\":" << json_quote(model.config.family)
            << ",\"task\":" << json_quote(engine::runtime::to_string(model.task.task))
            << ",\"mode\":" << json_quote(engine::runtime::to_string(model.task.mode))
            << "}";
    }
    out << "]}";
    return out.str();
}

}  // namespace minitts::server
