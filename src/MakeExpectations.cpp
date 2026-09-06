#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <set>
#include <string_view>
#include <unordered_map>

#include "BaselineSegmenter.h"

static const std::u32string dualJoinLetters = U"بتثجحخسشصضطظعغفقكلمنهيئى";
static const std::u32string rightJoinLetters = U"آاٱأإدذرزوؤءة";
static const std::u32string noJoiningLetters = U"ء";
static const std::u32string joinCausingLetters = U"ـ";
static const std::u32string otherCharacters = U"٠١٢٣٤٥٦٧٨٩۝۞";

static std::u32string decodeUtf8(std::string_view text) {
  std::u32string result;
  for (size_t i = 0; i < text.size();) {
    const unsigned char lead = static_cast<unsigned char>(text[i++]);
    char32_t codePoint = 0;
    int continuationCount = 0;
    if (lead < 0x80) {
      codePoint = lead;
    } else if ((lead & 0xE0) == 0xC0) {
      codePoint = lead & 0x1F;
      continuationCount = 1;
    } else if ((lead & 0xF0) == 0xE0) {
      codePoint = lead & 0x0F;
      continuationCount = 2;
    } else if ((lead & 0xF8) == 0xF0) {
      codePoint = lead & 0x07;
      continuationCount = 3;
    } else {
      throw std::runtime_error("Invalid UTF-8 lead byte");
    }
    if (i + continuationCount > text.size())
      throw std::runtime_error("Truncated UTF-8 sequence");
    for (int j = 0; j < continuationCount; ++j) {
      const unsigned char next = static_cast<unsigned char>(text[i++]);
      if ((next & 0xC0) != 0x80)
        throw std::runtime_error("Invalid UTF-8 continuation byte");
      codePoint = (codePoint << 6) | (next & 0x3F);
    }
    result.push_back(codePoint);
  }
  return result;
}

static std::string encodeUtf8(const std::u32string& text) {
  std::string result;
  for (const char32_t codePoint : text) {
    if (codePoint <= 0x7F) {
      result.push_back(static_cast<char>(codePoint));
    } else if (codePoint <= 0x7FF) {
      result.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
      result.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else if (codePoint <= 0xFFFF) {
      result.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
      result.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
      result.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else {
      result.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
      result.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
      result.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
      result.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
  }
  return result;
}

enum class ArabicJoiningType { ISOL,
                               FINA,
                               MEDI,
                               INIT,
                               NONE };

struct UnicodeProp {
  unsigned int nb_paths[5];
  UnicodeProp(unsigned int value = 0) {
    for (int i = 0; i < 5; i++) nb_paths[i] = value;
  }
  UnicodeProp(unsigned int isol, unsigned int init, unsigned int medi, unsigned int fina) {
    nb_paths[(int)ArabicJoiningType::ISOL] = isol;
    nb_paths[(int)ArabicJoiningType::INIT] = init;
    nb_paths[(int)ArabicJoiningType::MEDI] = medi;
    nb_paths[(int)ArabicJoiningType::FINA] = fina;
    nb_paths[(int)ArabicJoiningType::NONE] = isol;
  }
};

static std::unordered_map<char32_t, UnicodeProp> initUnicodProp() {
  std::unordered_map<char32_t, UnicodeProp> charProps;

  charProps[0X0000] = {0};           // notdef
  charProps[0X0001] = {0};           // null
  charProps[0X000A] = {0};           // linefeed
  charProps[0X200D] = {0};           // zwj
  charProps[0X034F] = {0};           // cgj
  charProps[0X0020] = {0};           // space
  charProps[0X06DE] = {8};           // rubelhizb
  charProps[0X06E9] = {2};           // placeofsajdah
  charProps[0X06DD] = {0};           // endofaya
  charProps[0X064E] = {1};           // fatha
  charProps[0X0650] = {1};           // kasra
  charProps[0X064B] = {2};           // fathatan
  charProps[0X064D] = {2};           // kasratan
  charProps[0X08F0] = {2};           // fathatanidgham
  charProps[0X08F2] = {2};           // kasratanidgham
  charProps[0X064F] = {1};           // damma
  charProps[0X08F1] = {2};           // dammatanidgham
  charProps[0X064C] = {2};           // dammatan
  charProps[0X0651] = {1};           // shadda
  charProps[0X06E1] = {1};           // headkhah
  charProps[0X0652] = {1};           // sukun
  charProps[0X0653] = {1};           // maddahabove
  charProps[0X0670] = {1};           // smallalef
  charProps[0X06E2] = {1};           // meemiqlab
  charProps[0X06ED] = {1};           // smalllowmeem
  charProps[0X06E5] = {1};           // smallwaw
  charProps[0X06E6] = {1};           // smallyeh
  charProps[0X06E7] = {1};           // smallhighyeh
  charProps[0X08F3] = {1};           // smallhighwaw
  charProps[0X06DF] = {1};           // smallhighroundedzero
  charProps[0X06E0] = {1};           // rectangularzero
  charProps[0X06EC] = {1};           // roundedfilledhigh
  charProps[0X065C] = {1};           // roundedfilledlow
  charProps[0X0654] = {1};           // hamzaabove
  charProps[0X0655] = {1};           // hamzabelow
  charProps[0X06D9] = {1};           // waqf.lam
  charProps[0X06D7] = {2};           // waqf.qaf
  charProps[0X06DA] = {2};           // waqf.jeem
  charProps[0X06D6] = {1};           // waqf.sad
  charProps[0X06DB] = {3};           // waqf.smallhighthreedots
  charProps[0X06D8] = {1};           // waqf.meem
  charProps[0X06DC] = {1};           // smallhighseen
  charProps[0X06E3] = {1};           // smalllowseen
  charProps[0X06E8] = {2};           // smallhighnoon
  charProps[0X0621] = {1};           // hamza.isol
  charProps[0X0627] = {1};           // alef.isol
  charProps[0X0633] = {1};           // seen.isol
  charProps[0X0644] = {1};           // lam.isol
  charProps[0X0645] = {1};           // meem.isol
  charProps[0X0646] = {2};           // noon.isol
  charProps[0X0642] = {2};           // qaf.isol
  charProps[0X0641] = {2};           // feh.isol
  charProps[0X062D] = {1};           // hah.isol
  charProps[0X062F] = {1};           // dal.isol
  charProps[0X0631] = {1};           // reh.isol
  charProps[0X0648] = {1};           // waw.isol
  charProps[0X0647] = {1};           // heh.isol
  charProps[0X0635] = {1};           // sad.isol
  charProps[0X0639] = {1};           // ain.isol
  charProps[0X0637] = {1};           // tah.isol
  charProps[0X0649] = {1};           // alefmaksura.isol
  charProps[0X064A] = {1, 2, 2, 1};  // yehshape.isol
  charProps[0X0643] = {2, 1, 1, 2};  // kaf.isol
  charProps[0X0660] = {1};           // zeroindic
  charProps[0X0661] = {1};           // oneindic
  charProps[0X0662] = {1};           // twoindic
  charProps[0X0663] = {1};           // threeindic
  charProps[0X0664] = {1};           // fourindic
  charProps[0X0665] = {1};           // fiveindic
  charProps[0X0666] = {1};           // sixindic
  charProps[0X0667] = {1};           // sevenindic
  charProps[0X0668] = {1};           // eightindic
  charProps[0X0669] = {1};           // nineindic
  charProps[0X0640] = {1};           // tatweel

  // not declared

  charProps[0x0622] = {2};  // alef.maddahabove.isol
  charProps[0x0623] = {2};  // alef.hamzaabove.isol
  charProps[0x0624] = {2};  // waw.hamzaabove.isol"
  charProps[0x0625] = {2};  // alef.hamzabelow.isol
  charProps[0x0626] = {2};  // alefmaksura.hamzaabove.isol
  charProps[0x0628] = {2};  // behshape.onedotdown.isol
  charProps[0x0629] = {2};  // heh.twodotsup.isol
  charProps[0x062A] = {2};  // behshape.twodotsup.isol
  charProps[0x062B] = {3};  // behshape.three_dots.isol
  charProps[0x062C] = {2};  // hah.onedotdown.isol
  charProps[0x062E] = {2};  // hah.onedotup.isol
  charProps[0x0630] = {2};  // dal.onedotup.isol
  charProps[0x0632] = {2};  // reh.onedotup.isol
  charProps[0x0634] = {3};  // seen.three_dots.isol
  charProps[0x0636] = {2};  // sad.onedotup.isol
  charProps[0x0638] = {2};  // tah.onedotup.isol
  charProps[0x063A] = {2};  // ain.onedotup.isol
  charProps[0x0671] = {2};  // alef.wasla.isol

  for (char32_t character : otherCharacters) {
    charProps[character] = {1};
  }
  // Aya
  charProps[0x06DD] = {2};

  return charProps;
}

std::unordered_map<char32_t, UnicodeProp> charProps = initUnicodProp();

static std::set<char32_t> initBases() {
  std::set<char32_t> bases;
  for (char32_t character : dualJoinLetters) {
    bases.insert(character);
  }
  for (char32_t character : rightJoinLetters) {
    bases.insert(character);
  }
  for (char32_t character : noJoiningLetters) {
    bases.insert(character);
  }
  for (char32_t character : joinCausingLetters) {
    bases.insert(character);
  }
  for (char32_t character : otherCharacters) {
    bases.insert(character);
  }
  return bases;
}

static std::set<char32_t> bases = initBases();

std::vector<WordInfo> BaselineSegmenter::makeExpectations(const LineLayout& lineLayout) {
  std::vector<WordInfo> wordInfos;
  wordInfos.reserve(lineLayout.words.size());

  for (const WordLayout& wordLayout : lineLayout.words) {
    WordInfo wordInfo;
    wordInfo.text = wordLayout.text;

    char32_t prevBase = 0;
    for (char32_t character : decodeUtf8(wordInfo.text)) {
      if (bases.contains(character)) {
        wordInfo.baseText += character;
        if (prevBase == 0 || noJoiningLetters.contains(character) ||
            rightJoinLetters.contains(prevBase) ||
            noJoiningLetters.contains(prevBase) ||
            otherCharacters.contains(prevBase)) {
          wordInfo.subWords.push_back({});
        }
        auto& subWord = wordInfo.subWords.back();
        subWord.baseText += character;
        subWord.text += character;
        prevBase = character;
      } else {
        if (wordInfo.subWords.empty()) wordInfo.subWords.push_back({});
        auto& subWord = wordInfo.subWords.back();
        subWord.text += character;
        const int pathCount =
            charProps[character].nb_paths[(int)ArabicJoiningType::NONE];
        subWord.expMarkPaths += pathCount;
        const MarkClass markClass = combiningMarkClass(character);
        if (pathCount > 0) {
          ++subWord.expectedMarks[static_cast<std::size_t>(markClass)];
          const int physicalComponentCount =
              markClass == MarkClass::QuranicSign && pathCount > 1
                  ? pathCount
                  : 0;
          subWord.expectedMarkInstances.push_back(
              {markClass, static_cast<int>(subWord.baseText.size()) - 1,
               physicalComponentCount});
        }
        if (isSmallLetterClass(markClass))
          subWord.expSmallLetterPaths += pathCount;
      }
    }

    for (auto& subWord : wordInfo.subWords) {
      subWord.utf8Text = encodeUtf8(subWord.text);
      subWord.baseCount = static_cast<int>(subWord.baseText.size());
      for (int i = 0; i < subWord.baseText.size(); i++) {
        ArabicJoiningType pos;
        if (subWord.baseText.size() == 1) {
          pos = ArabicJoiningType::ISOL;
        } else if (i == 0) {
          pos = ArabicJoiningType::INIT;
        } else if (i == subWord.baseText.size() - 1) {
          pos = ArabicJoiningType::FINA;
        } else {
          pos = ArabicJoiningType::MEDI;
        }
        auto nbPath =
            (int)charProps[subWord.baseText.at(i)].nb_paths[(int)pos];
        subWord.expBodyConn++;
        if (pos == ArabicJoiningType::FINA || pos == ArabicJoiningType::MEDI) {
          --subWord.expBodyConn;
        }
        subWord.expMarkPaths += std::max(0, nbPath - 1);
        if (nbPath > 1) {
          const MarkClass dotClass = baseDotClass(subWord.baseText.at(i));
          if (dotClass != MarkClass::Other) {
            ++subWord.expectedMarks[static_cast<std::size_t>(dotClass)];
            subWord.expectedMarkInstances.push_back({dotClass, i});
          } else {
            // Some Arabic glyphs have a detached second path that is part of
            // the base glyph rather than a semantic mark.  The final Kaf is
            // the important case here: its small upper contour can look very
            // similar to hamza-above.  Keep an explicit untyped slot so the
            // PAW DP can assign that contour to Other instead of forcing it
            // into one of the combining-mark classes.
            subWord.expectedMarkInstances.push_back(
                {MarkClass::Other, i});
          }
        }
      }

      assert(subWord.expBodyConn == 1);

      wordInfo.expMarkPaths += subWord.expMarkPaths;
      wordInfo.expSmallLetterPaths += subWord.expSmallLetterPaths;
      for (std::size_t mark = 0; mark < kMarkClassCount; ++mark)
        wordInfo.expectedMarks[mark] += subWord.expectedMarks[mark];
      wordInfo.expBodyConn += subWord.expBodyConn;
    }

    assert(wordInfo.subWords.size() == wordInfo.expBodyConn);

    wordInfos.emplace_back(std::move(wordInfo));
  }

  return wordInfos;
}
