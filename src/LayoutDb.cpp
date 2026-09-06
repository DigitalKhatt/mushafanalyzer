#include "LayoutDb.h"

#include <memory>
#include <stdexcept>

namespace {
struct StatementDeleter {
  void operator()(sqlite3_stmt* statement) const { sqlite3_finalize(statement); }
};
using Statement = std::unique_ptr<sqlite3_stmt, StatementDeleter>;

std::string columnText(sqlite3_stmt* statement, int column) {
  const auto* value = sqlite3_column_text(statement, column);
  return value ? reinterpret_cast<const char*>(value) : std::string{};
}
}  // namespace

void loadLayout(sqlite3* db, std::map<int, PageLayout>& pages) {
  pages.clear();
  if (!db) throw std::runtime_error("Database is not open.");

  constexpr const char* sql =
      "select l.page, l.line, w.dk_v1 as text, l.type "
      "from qpc_v1_layout l "
      "LEFT JOIN words w "
      "  ON l.type = 'ayah' "
      " AND l.range_start <= w.word_number_all "
      " AND l.range_end >= w.word_number_all "
      "ORDER BY l.page, l.line, w.word_number_all;";

  sqlite3_stmt* rawStatement = nullptr;
  if (sqlite3_prepare_v2(db, sql, -1, &rawStatement, nullptr) != SQLITE_OK)
    throw std::runtime_error(sqlite3_errmsg(db));
  Statement statement(rawStatement);

  while (sqlite3_step(statement.get()) == SQLITE_ROW) {
    const int page = sqlite3_column_int(statement.get(), 0);
    const int line = sqlite3_column_int(statement.get(), 1);
    const std::string text = columnText(statement.get(), 2);
    const std::string type = columnText(statement.get(), 3);
    auto [it, inserted] = pages.try_emplace(page, PageLayout{page, {}});
    (void)inserted;
    PageLayout& pageLayout = it->second;
    while (static_cast<int>(pageLayout.lines.size()) < line)
      pageLayout.lines.emplace_back();
    LineLayout& lineLayout = pageLayout.lines.at(line - 1);
    lineLayout.page = page;
    lineLayout.line = line;
    lineLayout.type = type;
    lineLayout.words.push_back({text});
  }
}
