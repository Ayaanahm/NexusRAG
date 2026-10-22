import type { Metadata, Viewport } from "next";
import { JetBrains_Mono, Literata, Public_Sans } from "next/font/google";
import "./globals.css";

const ui = Public_Sans({ subsets: ["latin"], variable: "--font-ui", display: "swap" });
const reading = Literata({ subsets: ["latin"], variable: "--font-reading", display: "swap" });
const code = JetBrains_Mono({ subsets: ["latin"], variable: "--font-code", display: "swap" });

export const metadata: Metadata = {
  title: "NexusRAG",
  description: "Cited answers from your documentation, with hybrid search you can inspect.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${ui.variable} ${reading.variable} ${code.variable}`}>
      <body>{children}</body>
    </html>
  );
}
