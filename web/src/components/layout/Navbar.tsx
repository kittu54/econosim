"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Landmark, BookOpen, BarChart3 } from "lucide-react";
import clsx from "clsx";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: BarChart3 },
  { href: "/docs", label: "Documentation", icon: BookOpen },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-50 glass-strong border-b border-border/40">
      <div className="flex items-center h-12 px-4">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-2.5 mr-8 group">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent via-accent-2 to-accent-3 flex items-center justify-center shadow-md group-hover:shadow-accent/20 transition-shadow">
            <Landmark className="w-3.5 h-3.5 text-white" />
          </div>
          <span className="text-sm font-bold tracking-tight text-foreground">
            EconoSim
          </span>
        </Link>

        {/* Nav links */}
        <div className="flex items-center gap-1">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const isActive = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={clsx(
                  "flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-200",
                  isActive
                    ? "text-accent bg-accent/10"
                    : "text-muted hover:text-foreground hover:bg-surface-2/50"
                )}
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
              </Link>
            );
          })}
        </div>

        {/* Right side — version badge */}
        <div className="ml-auto flex items-center gap-3">
          <span className="text-[10px] text-muted-2 font-mono px-2 py-0.5 rounded-full border border-border/60 bg-surface/40">
            v0.5 — 494 tests
          </span>
        </div>
      </div>
    </nav>
  );
}
