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
}

export default function KpiCard({
  label,
  value,
  delta,
  deltaDirection = "neutral",
  invertDelta = false,
  icon,
  accentColor = "from-blue-500/20 to-blue-600/5",
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
        "relative overflow-hidden rounded-xl border border-border",
        "bg-gradient-to-br from-surface to-surface-2",
        "p-5 transition-all duration-300",
        "hover:border-border-2 hover:shadow-lg hover:shadow-accent/5"
      )}
    >
      <div
        className={clsx(
          "absolute inset-0 bg-gradient-to-br opacity-40",
          accentColor
        )}
      />
      <div className="relative">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted">
            {label}
          </span>
          {icon && <span className="text-muted">{icon}</span>}
        </div>
        <div className="text-2xl font-bold tracking-tight text-foreground">
          {value}
        </div>
        {delta && (
          <div
            className={clsx("flex items-center gap-1 mt-2 text-xs font-medium", {
              "text-emerald-400": isPositive,
              "text-rose-400": isNegative,
              "text-muted": deltaDirection === "neutral",
            })}
          >
            {deltaDirection === "up" && <TrendingUp className="w-3 h-3" />}
            {deltaDirection === "down" && <TrendingDown className="w-3 h-3" />}
            {deltaDirection === "neutral" && <Minus className="w-3 h-3" />}
            <span>{delta}</span>
          </div>
        )}
      </div>
    </div>
  );
}
