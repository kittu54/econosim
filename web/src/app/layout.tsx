import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/layout/Navbar";

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
      <body className="antialiased bg-background text-foreground min-h-screen relative">
        <Navbar />
        {children}
      </body>
    </html>
  );
}
