#pragma once

#include <array>
#include <cstddef>
#include <string_view>

enum class MarkClass : std::size_t {
  DotAboveOne,
  DotBelowOne,
  DotAboveTwo,
  DotBelowTwo,
  DotAboveThree,
  Fatha,
  Kasra,
  Damma,
  Fathatan,
  Kasratan,
  Dammatan,
  Shadda,
  Sukun,
  Maddah,
  HamzaAbove,
  HamzaBelow,
  SmallAlef,
  SmallYeh,
  SmallWaw,
  SmallMeem,
  SmallNoon,
  SmallSeen,
  Wasla,
  WaqfLam,
  WaqfQaf,
  WaqfJeem,
  WaqfSad,
  WaqfMeem,
  WaqfThreeDots,
  QuranicSign,
  Other,
  Count
};

constexpr std::size_t kMarkClassCount =
    static_cast<std::size_t>(MarkClass::Count);
using MarkInventory = std::array<int, kMarkClassCount>;

std::string_view markClassName(MarkClass markClass);
MarkClass combiningMarkClass(char32_t character);
MarkClass baseDotClass(char32_t character);
bool isSmallLetterClass(MarkClass markClass);
