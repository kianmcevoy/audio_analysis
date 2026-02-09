#pragma once
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <algorithm>

struct AnalysisRecorder
{
    void set_path(const std::filesystem::path& path)
    {
        output_dir = path;
    }
    
    void begin(int length_blocks, int sr)
    {
        remaining = length_blocks;
        total = length_blocks;
        sample_rate = sr;
        active = true;
        taps.clear();
        finished = false;
    }

    inline void capture(const char* name, float l, float r)
    {
        if (!active) return;
        auto& v = taps[name];
        v.push_back(l);
        v.push_back(r);
    }

    inline bool tick()
    {
        if (!active) return false;
        if (--remaining <= 0) 
        {
            active = false;
            finished = true;
            return true;
        }
        return false;
    }  

    inline int16_t float_to_pcm16(float x)
    {
        x = std::max(-1.0f, std::min(1.0f, x));
        return int16_t(x * 32767.0f);
    }

    inline void write_wav(const std::filesystem::path& path, const std::vector<float>& stereo)
    {
        const uint32_t frames = stereo.size() / 2;
        const uint32_t data_bytes = frames * 4;

        std::ofstream f(path, std::ios::binary);
        f.write("RIFF", 4);
        uint32_t riff_size = 36 + data_bytes;
        f.write((char*)&riff_size, 4);
        f.write("WAVEfmt ", 8);

        uint32_t fmt_size = 16;
        uint16_t audio_fmt = 1;
        uint16_t chans = 2;
        uint32_t byte_rate = sample_rate * 4;
        uint16_t block_align = 4;
        uint16_t bits = 16;

        f.write((char*)&fmt_size, 4);
        f.write((char*)&audio_fmt, 2);
        f.write((char*)&chans, 2);
        f.write((char*)&sample_rate, 4);
        f.write((char*)&byte_rate, 4);
        f.write((char*)&block_align, 2);
        f.write((char*)&bits, 2);

        f.write("data", 4);
        f.write((char*)&data_bytes, 4);

        for (size_t i = 0; i < stereo.size(); i += 2) {
            int16_t l = float_to_pcm16(stereo[i]);
            int16_t r = float_to_pcm16(stereo[i + 1]);
            f.write((char*)&l, 2);
            f.write((char*)&r, 2);
        }
    }

    inline std::string timestamp()
    {
        std::time_t t = std::time(nullptr);
        std::tm tm{};
        localtime_r(&t, &tm);
        std::ostringstream ss;
        ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        return ss.str();
    }

    inline void write_bundle()
    {
        namespace fs = std::filesystem;
        fs::path root = output_dir / timestamp();
        fs::create_directories(root / "taps");

        for (const auto& [name, data] : taps) 
        {
            write_wav(root / "taps" / (name + ".wav"), data);
        }

        std::ofstream meta(root / "meta.json");
        meta << "{\n";
        meta << "  \"sample_rate_hz\": " << sample_rate << ",\n";
        meta << "  \"length_samples\": " << total << ",\n";
        meta << "  \"taps\": [";
        bool first = true;
        for (const auto& [name, _] : taps) 
        {
            if (!first) meta << ", ";
            meta << "\"" << name << "\"";
            first = false;
        }
        meta << "]\n}\n";
    }

    bool active = false;
    int remaining = 0;
    int total = 0;
    int sample_rate = 48000;
    std::filesystem::path output_dir;
    
    bool finished = false;

    std::map<std::string, std::vector<float>> taps; // interleaved L,R
};

#define ANALYSE_TAP(rec, name, l, r)     do { if ((rec).active) (rec).capture(name, l, r); } while (0)
