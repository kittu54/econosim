"use client";

export function KpiSkeleton() {
  return (
    <div className="rounded-xl border border-border/60 bg-surface/60 p-4">
      <div className="skeleton h-3 w-16 mb-3" />
      <div className="skeleton h-6 w-24 mb-2" />
      <div className="skeleton h-3 w-14" />
    </div>
  );
}

export function ChartSkeleton() {
  return (
    <div className="rounded-xl border border-border/60 bg-surface/60 p-4">
      <div className="skeleton h-4 w-40 mb-4" />
      <div className="skeleton h-[260px] w-full rounded-lg" />
    </div>
  );
}

export function DashboardSkeleton() {
  return (
    <div className="p-6 space-y-6 animate-fade-in">
      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <KpiSkeleton key={i} />
        ))}
      </div>
      {/* Tab bar skeleton */}
      <div className="flex gap-2 pb-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="skeleton h-9 w-28 rounded-lg" />
        ))}
      </div>
      {/* Chart grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartSkeleton />
        <ChartSkeleton />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartSkeleton />
        <ChartSkeleton />
      </div>
    </div>
  );
}
