export function fmtNumber(n: number | undefined, decimals = 0): string {
  if (n === undefined || n === null || isNaN(n)) return "—";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function fmtPercent(n: number | undefined, decimals = 1): string {
  if (n === undefined || n === null || isNaN(n)) return "—";
  return `${(n * 100).toFixed(decimals)}%`;
}

export function fmtDelta(current: number, initial: number): string {
  const diff = current - initial;
  const sign = diff >= 0 ? "+" : "";
  return `${sign}${fmtNumber(diff)}`;
}

export function fmtDeltaPercent(current: number, initial: number): string {
  const diff = current - initial;
  const sign = diff >= 0 ? "+" : "";
  return `${sign}${(diff * 100).toFixed(1)}%`;
}
