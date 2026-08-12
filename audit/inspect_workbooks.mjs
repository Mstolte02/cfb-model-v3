import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const files = process.argv.slice(2);
for (const path of files) {
  const book = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
  const summary = await book.inspect({
    kind: "workbook,sheet,table",
    maxChars: 12000,
    tableMaxRows: 5,
    tableMaxCols: 10,
    tableMaxCellChars: 80,
  });
  const formulas = await book.inspect({
    kind: "formula",
    maxChars: 5000,
    options: { maxResults: 100 },
  });
  process.stdout.write(`### ${path}\n${summary.ndjson}\n### formulas\n${formulas.ndjson}\n`);
}
