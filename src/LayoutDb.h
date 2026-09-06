#pragma once
#include <sqlite3.h>

#include <map>
#include <string>
#include <vector>

struct WordLayout {
  std::string text;
};

struct LineLayout {
  int page = -1;
  int line = -1;
  std::string type;
  std::vector<WordLayout> words;
};

struct PageLayout {
  int page = -1;
  std::vector<LineLayout> lines;
};

void loadLayout(sqlite3* db, std::map<int, PageLayout>& pages);
