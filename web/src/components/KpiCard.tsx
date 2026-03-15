"use client";

import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import clsx from "clsx";

interface KpiCardProps {
  label: string;
  value: string;
  delta?: string;
  deltaDirection?: "up" | "down" | "neutral";
  invertDelta?: boolean;
  icon?: React.ReactNode;
  accentColor?: string;
  sparklineData?: number[];
  sparklineColor?: string;
}

function MiniSparkline({
  data,
  color = "#3b82f6",
  width = 80,
  height = 28,
}: {
  data: number[];
  color?: string;
  width?: number;
  height?: number;
}) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const padding = 2;

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * (width - padding * 2) + padding;
    const y = height - padding - ((v - min) / range) * (height - padding * 2);
    return `${x},${y}`;
  });

  const areaPoints = [
    `${padding},${height}`,
    ...points,
    `${width - padding},${height}`,
  ].join(" ");

  return (
    <svg width={width} height={height} className="opacity-60">
      <defs>
        <linearGradient id={`spark-${color.replace("#", "")}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon
        points={areaPoints}
        fill={`url(#spark-${color.replace("#", "")})`}
      />
      <polyline
        points={points.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function KpiCard({
  label,
  value,
  delta,
  deltaDirection = "neutral",
  invertDelta = false,
  icon,
  accentColor = "from-blue-500/20 to-blue-600/5",
  sparklineData,
  sparklineColor,
}: KpiCardProps) {
  const isPositive = invertDelta
    ? deltaDirection === "down"
    : deltaDirection === "up";
  const isNegative = invertDelta
    ? deltaDirection === "up"
    : deltaDirection === "down";

  return (
    <div
      className={clsx(
        "relative overflow-hidden rounded-xl border border-border/80",
        "bg-surface/80 backdrop-blur-sm",
        "p-4 transition-all duration-300 group",
        "hover:border-border-2 hover:shadow-lg hover:shadow-accent/5",
        "hover:bg-surface"
      )}
    >
      {/* Subtle gradient overlay */}
      <div
        className={clsx(
          "absolute inset-0 bg-gradient-to-br opacity-30 transition-opacity group-hover:opacity-50",
          accentColor
        )}
      />
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">
            {label}
          </span>
          {icon && <span className="text-muted-2">{icon}</span>}
        </div>
        <div className="flex items-end justify-between gap-2">
          <div>
            <div className="text-xl font-bold tracking-tight text-foreground animate-count-up">
              {value}
            </div>
            {delta && (
              <div
                className={clsx(
                  "flex items-center gap-1 mt-1 text-[11px] font-medium",
                  {
                    "text-emerald-400": isPositive,
                    "text-rose-400": isNegative,
                    "text-muted": deltaDirection === "neutral",
                  }
                )}
              >
                {deltaDirection === "up" && <TrendingUp className="w-3 h-3" />}
                {deltaDirection === "down" && (
                  <TrendingDown className="w-3 h-3" />
                )}
                {deltaDirection === "neutral" && (
                  <Minus className="w-3 h-3" />
                )}
                <span>{delta}</span>
              </div>
            )}
          </div>
          {sparklineData && sparklineData.length > 2 && (
            <MiniSparkline
              data={sparklineData}
              color={sparklineColor || "#3b82f6"}
            />
          )}
        </div>
      </div>
    </div>
  );
}
