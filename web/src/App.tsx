function App() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 bg-amber-50 p-6 text-slate-800">
      <h1 className="text-4xl font-bold tracking-tight">Doodle Live</h1>
      <p className="max-w-md text-center text-slate-600">
        Draw something. An AI watches every stroke and guesses out loud. Coming soon.
      </p>
      <footer className="absolute bottom-4 text-xs text-slate-500">
        Drawings from{' '}
        <a
          className="underline hover:text-slate-700"
          href="https://quickdraw.withgoogle.com/data"
          target="_blank"
          rel="noreferrer"
        >
          Quick, Draw!
        </a>{' '}
        (CC BY 4.0)
      </footer>
    </main>
  )
}

export default App
