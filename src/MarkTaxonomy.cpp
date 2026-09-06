#include "MarkTaxonomy.h"

#include <array>

std::string_view markClassName(MarkClass markClass) {
  static constexpr std::array<std::string_view, kMarkClassCount> names = {
      "dot_above_one", "dot_below_one", "dot_above_two",
      "dot_below_two", "dot_above_three", "fatha",
      "kasra", "damma", "fathatan",
      "kasratan", "dammatan", "shadda",
      "sukun", "maddah", "hamza_above",
      "hamza_below", "small_alef", "small_yeh",
      "small_waw", "small_meem", "small_noon",
      "small_seen", "wasla", "waqf_lam", "waqf_qaf",
      "waqf_jeem", "waqf_sad", "waqf_meem",
      "waqf_three_dots", "quranic_sign", "other"};
  return names.at(static_cast<std::size_t>(markClass));
}

MarkClass combiningMarkClass(char32_t character) {
  switch (character) {
    case 0x064E:
      return MarkClass::Fatha;
    case 0x0650:
      return MarkClass::Kasra;
    case 0x064F:
      return MarkClass::Damma;
    case 0x064B:
    case 0x08F0:
      return MarkClass::Fathatan;
    case 0x064D:
    case 0x08F2:
      return MarkClass::Kasratan;
    case 0x064C:
    case 0x08F1:
      return MarkClass::Dammatan;
    case 0x0651:
      return MarkClass::Shadda;
    case 0x0652:
    case 0x06E1:
      return MarkClass::Sukun;
    case 0x0653:
      return MarkClass::Maddah;
    case 0x0654:
      return MarkClass::HamzaAbove;
    case 0x0655:
      return MarkClass::HamzaBelow;
    case 0x0670:
      return MarkClass::SmallAlef;
    case 0x06E6:
    case 0x06E7:
      return MarkClass::SmallYeh;
    case 0x06E5:
    case 0x08F3:
      return MarkClass::SmallWaw;
    case 0x06E2:
    case 0x06ED:
      return MarkClass::SmallMeem;
    case 0x06E8:
      return MarkClass::SmallNoon;
    case 0x06DC:
    case 0x06E3:
      return MarkClass::SmallSeen;
    case 0x06D9:
      return MarkClass::WaqfLam;
    case 0x06D7:
      return MarkClass::WaqfQaf;
    case 0x06DA:
      return MarkClass::WaqfJeem;
    case 0x06D6:
      return MarkClass::WaqfSad;
    case 0x06D8:
      return MarkClass::WaqfMeem;
    case 0x06DB:
      return MarkClass::WaqfThreeDots;
    case 0x06DF:
    case 0x06E0:
    case 0x06E9:
    case 0x06EC:
      return MarkClass::QuranicSign;
    default:
      return MarkClass::Other;
  }
}

MarkClass baseDotClass(char32_t character) {
  switch (character) {
    case U'أ':
      return MarkClass::HamzaAbove;
    case U'إ':
      return MarkClass::HamzaBelow;
    case U'آ':
      return MarkClass::Maddah;
    case U'ٱ':
      return MarkClass::Wasla;
    case U'ب':
    case U'ج':
    case U'خ':
    case U'ذ':
    case U'ز':
    case U'ض':
    case U'ظ':
    case U'غ':
    case U'ف':
    case U'ن':
      return character == U'ب' || character == U'ج'
                 ? MarkClass::DotBelowOne
                 : MarkClass::DotAboveOne;
    case U'ت':
    case U'ق':
    case U'ة':
      return MarkClass::DotAboveTwo;
    case U'ي':
    case U'ى':
      return MarkClass::DotBelowTwo;
    case U'ث':
    case U'ش':
      return MarkClass::DotAboveThree;
    default:
      return MarkClass::Other;
  }
}

bool isSmallLetterClass(MarkClass markClass) {
  return markClass >= MarkClass::SmallAlef &&
         markClass <= MarkClass::SmallSeen;
}
