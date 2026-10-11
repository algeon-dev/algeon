const GPU_CALL = /\b(?:cugraph|cuvs|cuml|cuopt)_\w+\s*\(/;

export function gpuCallMetastring(code: string, metastring?: string): string | undefined {
  if (metastring?.includes('{')) {
    return metastring;
  }
  const codeLines = code.split('\n');
  const lines = codeLines.flatMap((line, index) => (GPU_CALL.test(line) ? [index + 1] : []));
  // A highlight on every non-blank line, as in a one-line call, separates
  // nothing from the surrounding SQL.
  const marksSubset = lines.length < codeLines.filter(line => line.trim() !== '').length;
  return lines.length > 0 && marksSubset
    ? [`{${lines.join(',')}}`, metastring].filter(Boolean).join(' ')
    : metastring;
}
