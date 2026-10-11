import type {CSSProperties} from 'react';

export const bytesPerGiB = 1024 ** 3;

export function formatGiB(value: number): string {
  return `${(value / bytesPerGiB).toFixed(1)} GiB`;
}

export function formatGiBShort(value: number): string {
  return `${value.toFixed(1)} GiB`;
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(value);
}

// Recharts defaults to a white tooltip; Infima variables keep it readable in both themes.
export const tooltipContentStyle: CSSProperties = {
  background: 'var(--ifm-background-surface-color)',
  border: '1px solid var(--ifm-color-emphasis-300)',
  borderRadius: '6px',
  boxShadow: 'var(--ifm-global-shadow-md)',
  color: 'var(--ifm-font-color-base)',
  fontSize: '0.82rem',
  lineHeight: 1.4,
  padding: '0.5rem 0.75rem',
};

export const tooltipLabelStyle: CSSProperties = {
  color: 'var(--ifm-heading-color)',
  fontWeight: 600,
  marginBottom: '0.25rem',
};

export const tooltipItemStyle: CSSProperties = {
  color: 'var(--ifm-font-color-base)',
  padding: 0,
};

export const tooltipCursorStyle = {
  fill: 'var(--ifm-color-emphasis-300)',
  fillOpacity: 0.35,
} as const;
