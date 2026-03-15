import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EconoSim — Multi-Agent Economic Simulation",
  description:
    "Interactive dashboard for a multi-agent AI economic simulation with stock-flow consistent accounting.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}
