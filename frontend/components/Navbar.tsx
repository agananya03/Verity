"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function Navbar() {
  const pathname = usePathname();

  const navItems = [
    {
      name: "Chat",
      href: "/",
    },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#0F1419]/95 backdrop-blur border-b border-[#2A3138]">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
        {/* Brand / Logo - Filing Cover Page Masthead */}
        <Link href="/" className="flex flex-col group focus-visible:outline-none">
          <span className="font-display font-semibold text-xl tracking-tight text-[#EDE7DA] group-hover:text-[#E8A33D] transition-colors">
            VERITY
          </span>
          <span className="font-mono text-[11px] text-[#8B95A1] tracking-normal">
            SEC Filing Intelligence · Form 10-K Research
          </span>
        </Link>

        {/* Navigation Links - Mono, Understated, Active Underline in --highlight */}
        <nav className="flex items-center gap-6 font-mono text-xs sm:text-sm">
          {navItems.map((item) => {
            const isActive = pathname === item.href;

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`relative py-1 transition-colors ${
                  isActive
                    ? "text-[#EDE7DA] font-medium"
                    : "text-[#8B95A1] hover:text-[#EDE7DA]"
                }`}
              >
                {item.name}
                {isActive && (
                  <span className="absolute left-0 bottom-0 w-full h-[2px] bg-[#E8A33D]" />
                )}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
