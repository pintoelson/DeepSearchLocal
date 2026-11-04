import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeepSearch-Local | Autonomous Deep Research Agent",
  description: "Offline-capable deep research agent running on consumer hardware",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
        {children}
      </body>
    </html>
  );
}
