#pragma once

#include <glaze/glaze.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

template <typename T>
void writeJsonFile(const std::filesystem::path& path, const T& value) {
  std::string json;
  if (const auto error = glz::write<glz::opts{.prettify = true}>(value, json))
    throw std::runtime_error("Could not serialize JSON: " +
                             glz::format_error(error, json));
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("Could not open " + path.string());
  output << json;
  if (!output) throw std::runtime_error("Could not write " + path.string());
}
