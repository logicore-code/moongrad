// Run the browser example's WebAssembly module under Node.
//
// The page does the same thing with the same shim; this script exists
// so that the module is known to work before anyone opens a browser.
//
//   moon build --target wasm examples/browser
//   node examples/browser/run.mjs

import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { runWasm } from './wasi.js'

const here = dirname(fileURLToPath(import.meta.url))
const bytes = await readFile(
  join(here, '..', '..', '_build', 'wasm', 'debug', 'build', 'examples',
    'browser', 'browser.wasm'),
)
const text = await runWasm(bytes)
process.stdout.write(text)
