import type { Metadata } from "next";
import { Fraunces, Inter, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

const fontFraunces = Fraunces({
  subsets: ["latin"],
  weight: ["500", "600"],
  variable: "--font-fraunces",
});

const fontInter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const fontMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Verity | SEC Filing Financial RAG Intelligence",
  description:
    "Agentic financial RAG assistant with self-correcting audit verification and benchmark evaluation dashboard.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${fontFraunces.variable} ${fontInter.variable} ${fontMono.variable} font-body bg-[#0F1419] text-[#EDE7DA] antialiased min-h-screen flex flex-col`}>
        <Navbar />
        <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>
      </body>
    </html>
  );
}
