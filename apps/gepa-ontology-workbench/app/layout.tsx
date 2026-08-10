import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GEPA Ontology Workbench",
  description: "Explore, assess, and export invariant-governed GEPA Mindfulness ontology bundles.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
