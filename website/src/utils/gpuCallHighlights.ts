const GPU_CALL = /\b(?:cugraph|cuvs|cuml|cuopt)_\w+\s*\(/;

export function gpuCallMetastring(code: string, metastring?: string): string | undefined {
  if (metastring?.includes('{')) {
    return metastring;
  }
  const lines = code
    .split('\n')
    .flatMap((line, index) => (GPU_CALL.test(line) ? [index + 1] : []));
  return lines.length > 0
    ? [`{${lines.join(',')}}`, metastring].filter(Boolean).join(' ')
    : metastring;
}
