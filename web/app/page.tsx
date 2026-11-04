export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <div className="max-w-2xl text-center space-y-4">
        <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl">
          DeepSearch<span className="text-cyan-400">-Local</span>
        </h1>
        <p className="text-zinc-400 text-lg">
          Autonomous deep research agent platform running entirely on consumer GPUs.
        </p>
        <div className="pt-4 text-sm text-zinc-500 border-t border-zinc-800">
          Scaffolding initialized. UI components and SSE streaming consumer to be wired in Phase 5.
        </div>
      </div>
    </main>
  );
}
