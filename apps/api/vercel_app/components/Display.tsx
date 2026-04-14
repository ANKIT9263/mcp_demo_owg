'use client'

interface DisplayProps {
  value: string
  history: string[]
}

export default function Display({ value, history }: DisplayProps) {
  return (
    <div className="bg-slate-950 p-6 border-b border-slate-700">
      <div className="text-right">
        <div className="text-sm text-neutral/60 mb-2 min-h-5">
          {history.length > 0 && history[history.length - 1]}
        </div>
        <div className="text-5xl font-bold text-foreground break-words">
          {value}
        </div>
      </div>
    </div>
  )
}
