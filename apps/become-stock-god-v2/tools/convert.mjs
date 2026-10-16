// Convert immutable AlphaPilot parquet sources into compact JSON cache.
// Source files are read-only; this tool writes only to BSG_CACHE (default: ./cache/).
import fs from 'node:fs'
import path from 'node:path'

const ROOT = process.cwd()
const HISTORY = process.env.BSG_HISTORY_DIR || path.join(ROOT, '..', 'alphapilot-market-data-v2', 'data', 'history', '2020-2025')
const OUT = process.env.BSG_CACHE || path.join(ROOT, 'cache')
fs.mkdirSync(OUT, { recursive: true })

const { parquetMetadataAsync, parquetRead, asyncBufferFromFile } = await import('hyparquet')

async function readCols(fileName, cols) {
  const file = await asyncBufferFromFile(path.join(HISTORY, fileName))
  const md = await parquetMetadataAsync(file)
  const n = Number(md.num_rows)
  const out = {}
  for (const c of cols) out[c] = new Array(n)
  await parquetRead({ file, metadata: md, columns: cols, onChunk: ({ columnName, columnData, rowStart }) => {
    const a = out[columnName]
    for (let i = 0; i < columnData.length; i++) a[rowStart + i] = columnData[i]
  } })
  return { n, out }
}
const conv = v => v === null || v === undefined ? null : (typeof v === 'bigint' ? Number(v) : v)

const rows = []
for (let y = 2020; y <= 2025; y++) {
  const { n, out } = await readCols(`ohlcv_${y}.parquet`, ['date','code','name','volume','open','high','low','close'])
  for (let i = 0; i < n; i++) rows.push([conv(out.date[i]), out.code[i], out.name[i], conv(out.volume[i]), out.open[i], out.high[i], out.low[i], out.close[i], y])
  console.log('ohlcv', y, n)
}
fs.writeFileSync(path.join(OUT, 'ohlcv.json'), JSON.stringify(rows))

const inst = await readCols('institutional_2020_2025.parquet', ['date','market','code','foreign_net','trust_net','dealer_net'])
const irows = []
for (let i = 0; i < inst.n; i++) irows.push([conv(inst.out.date[i]), inst.out.market[i], inst.out.code[i], inst.out.foreign_net[i], inst.out.trust_net[i], inst.out.dealer_net[i]])
fs.writeFileSync(path.join(OUT, 'inst.json'), JSON.stringify(irows))
console.log('inst', inst.n)
