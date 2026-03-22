"use client";

import { useState, useRef, useCallback } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ColumnSummary {
  name: string;
  dtype: string;
  unique: number;
  nullPct: number;
  nullCount: number;
}

interface DataProfile {
  filename: string;
  rows: number;
  columns: number;
  duplicates: number;
  columnSummary: ColumnSummary[];
  sample: string;
  stats: string;
  previewRows: Record<string, unknown>[];
  previewHeaders: string[];
}

const SUPPORTED_EXTENSIONS = [".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"];

async function parseFileToRows(file: File): Promise<Record<string, unknown>[]> {
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();

  if (ext === ".csv") {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true, skipEmptyLines: true,
        complete: (r) => resolve(r.data as Record<string, unknown>[]),
        error: (err: { message: string }) => reject(new Error(err.message)),
      });
    });
  }

  if (ext === ".tsv") {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true, skipEmptyLines: true, delimiter: "\t",
        complete: (r) => resolve(r.data as Record<string, unknown>[]),
        error: (err: { message: string }) => reject(new Error(err.message)),
      });
    });
  }

  if (ext === ".xlsx" || ext === ".xls") {
    const buffer = await file.arrayBuffer();
    const wb = XLSX.read(buffer);
    const ws = wb.Sheets[wb.SheetNames[0]];
    return XLSX.utils.sheet_to_json(ws) as Record<string, unknown>[];
  }

  if (ext === ".json") {
    const text = await file.text();
    const json = JSON.parse(text);
    if (Array.isArray(json)) return json as Record<string, unknown>[];
    for (const key of Object.keys(json)) {
      if (Array.isArray(json[key])) return json[key] as Record<string, unknown>[];
    }
    return [json] as Record<string, unknown>[];
  }

  if (ext === ".parquet") {
    const { parquetRead } = await import("hyparquet");
    const buffer = await file.arrayBuffer();
    const rows: Record<string, unknown>[] = [];
    await parquetRead({
      file: {
        byteLength: buffer.byteLength,
        slice: (start: number, end?: number) => Promise.resolve(buffer.slice(start, end)),
      },
      rowFormat: "object",
      onComplete: (data: Record<string, unknown>[]) => { rows.push(...data); },
    });
    return rows;
  }

  throw new Error(`Unsupported format: ${ext}`);
}

function inferDtype(values: unknown[]): string {
  const nonNull = values.filter((v) => v !== "" && v !== null && v !== undefined);
  if (nonNull.length === 0) return "empty";
  const numericCount = nonNull.filter((v) => !isNaN(Number(v))).length;
  return numericCount / nonNull.length > 0.9 ? "numeric" : "string";
}

function profileData(filename: string, data: Record<string, unknown>[]): DataProfile {
  if (data.length === 0) throw new Error("Empty CSV");
  const headers = Object.keys(data[0]);
  const rows = data.length;
  const rowStrs = data.map((r) => JSON.stringify(r));
  const duplicates = rows - new Set(rowStrs).size;

  const columnSummary: ColumnSummary[] = headers.map((col) => {
    const values = data.map((r) => r[col]);
    const nullCount = values.filter((v) => v === "" || v === null || v === undefined).length;
    const unique = new Set(values.filter((v) => v !== "" && v !== null)).size;
    return {
      name: col,
      dtype: inferDtype(values),
      unique,
      nullCount,
      nullPct: parseFloat(((nullCount / rows) * 100).toFixed(1)),
    };
  });

  const sample = [
    headers.join(" | "),
    headers.map(() => "---").join(" | "),
    ...data.slice(0, 5).map((r) => headers.map((h) => String(r[h] ?? "")).join(" | ")),
  ].join("\n");

  const statsLines: string[] = [];
  for (const col of columnSummary) {
    if (col.dtype === "numeric") {
      const nums = data.map((r) => Number(r[col.name])).filter((v) => !isNaN(v));
      if (!nums.length) continue;
      const sorted = [...nums].sort((a, b) => a - b);
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      const std = Math.sqrt(nums.reduce((a, b) => a + (b - mean) ** 2, 0) / nums.length);
      const median = sorted[Math.floor(sorted.length / 2)];
      statsLines.push(
        `${col.name}: min=${sorted[0].toFixed(2)}, max=${sorted[sorted.length-1].toFixed(2)}, mean=${mean.toFixed(2)}, median=${median.toFixed(2)}, std=${std.toFixed(2)}`
      );
    }
  }

  return {
    filename, rows, columns: headers.length, duplicates,
    columnSummary, sample,
    stats: statsLines.join("\n") || "No numeric columns",
    previewRows: data.slice(0, 10),
    previewHeaders: headers,
  };
}

export default function Home() {
  const [profile, setProfile] = useState<DataProfile | null>(null);
  const [question, setQuestion] = useState("");
  const [description, setDescription] = useState("");
  const [analysis, setAnalysis] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported format. Please upload: CSV, TSV, Excel, JSON, or Parquet.`);
      return;
    }
    setError(""); setAnalysis("");
    try {
      const rows = await parseFileToRows(file);
      setProfile(profileData(file.name, rows));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to parse file");
    }
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const runAnalysis = async () => {
    if (!profile) return;
    setError(""); setAnalysis(""); setLoading(true);
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile, question, description }),
      });
      if (!res.ok) { const err = await res.json(); throw new Error(err.error || "Request failed"); }
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let text = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        text += decoder.decode(value, { stream: true });
        setAnalysis(text);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center gap-3">
          <span className="text-2xl">🧠</span>
          <div>
            <h1 className="text-xl font-bold text-white">Data Science Assistant</h1>
            <p className="text-xs text-gray-400">AI-powered dataset analysis with Claude</p>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Dataset</label>
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${dragging ? "border-blue-500 bg-blue-500/10" : "border-gray-700 hover:border-gray-500"}`}
            >
              <div className="text-3xl mb-2">📂</div>
              <p className="text-sm text-gray-400">
                {profile
                  ? <span className="text-green-400 font-medium">✓ {profile.filename}</span>
                  : <>Drop file here or <span className="text-blue-400">browse</span></>}
              </p>
              {profile
                ? <p className="text-xs text-gray-500 mt-1">{profile.rows.toLocaleString()} rows × {profile.columns} cols</p>
                : (
                  <div className="mt-2 flex flex-wrap justify-center gap-1">
                    {["CSV", "TSV", "Excel", "JSON", "Parquet"].map((fmt) => (
                      <span key={fmt} className="px-2 py-0.5 rounded-full bg-gray-800 text-gray-400 text-xs font-medium border border-gray-700">{fmt}</span>
                    ))}
                  </div>
                )}
            </div>
            <input ref={fileInputRef} type="file" accept=".csv,.tsv,.xlsx,.xls,.json,.parquet" onChange={onFileChange} className="hidden" />
          </div>

          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Question (optional)</label>
            <textarea value={question} onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. predict customer churn" rows={2}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none" />
          </div>

          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <label className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Description (optional)</label>
            <textarea value={description} onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g. E-commerce orders. Target is 'churn'." rows={3}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none" />
          </div>

          <button onClick={runAnalysis} disabled={loading || !profile}
            className="w-full py-3 rounded-xl font-semibold text-sm transition-all bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed">
            {loading ? "Analyzing..." : "▶ Run Analysis"}
          </button>

          {error && <div className="bg-red-900/40 border border-red-700 rounded-lg p-3 text-sm text-red-300">{error}</div>}
        </div>

        <div className="lg:col-span-2 space-y-4">
          {profile && (
            <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
                <h2 className="font-semibold text-sm">Data Preview</h2>
                <div className="flex gap-4 text-xs text-gray-400">
                  <span>{profile.rows.toLocaleString()} rows</span>
                  <span>{profile.columns} columns</span>
                  {profile.duplicates > 0 && <span className="text-yellow-400">{profile.duplicates} duplicates</span>}
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-gray-800">
                      {profile.previewHeaders.map((h) => (
                        <th key={h} className="px-3 py-2 text-left text-gray-400 font-medium whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {profile.previewRows.map((row, i) => (
                      <tr key={i} className="border-t border-gray-800 hover:bg-gray-800/50">
                        {profile.previewHeaders.map((h) => (
                          <td key={h} className="px-3 py-2 text-gray-300 whitespace-nowrap max-w-[200px] truncate">{String(row[h] ?? "")}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="border-t border-gray-800 p-4">
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Column Overview</p>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {profile.columnSummary.map((col) => (
                    <div key={col.name} className="bg-gray-800 rounded-lg px-3 py-2">
                      <p className="text-xs font-medium text-white truncate">{col.name}</p>
                      <p className="text-xs text-gray-500">{col.dtype} · {col.unique} unique</p>
                      {col.nullPct > 0 && <p className="text-xs text-yellow-400">{col.nullPct}% missing</p>}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {(analysis || loading) && (
            <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
                <h2 className="font-semibold text-sm">AI Analysis</h2>
                {loading && <span className="text-xs text-blue-400 animate-pulse">● Streaming...</span>}
                {!loading && analysis && (
                  <button onClick={() => {
                    const blob = new Blob([analysis], { type: "text/markdown" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url; a.download = `${profile?.filename}_analysis.md`; a.click();
                  }} className="text-xs text-blue-400 hover:text-blue-300">⬇ Download</button>
                )}
              </div>
              <div className="p-6 prose prose-invert prose-sm max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis}</ReactMarkdown>
              </div>
            </div>
          )}

          {!profile && !analysis && (
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-12 text-center">
              <div className="text-5xl mb-4">📊</div>
              <h3 className="text-lg font-semibold text-white mb-2">Upload a dataset to get started</h3>
              <p className="text-sm text-gray-400">Drop any CSV file and get a full AI-powered data science analysis in seconds.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
