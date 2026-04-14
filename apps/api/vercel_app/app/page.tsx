'use client'

import { useState } from 'react'
import Calculator from '@/components/Calculator'

export default function Home() {
  return (
    <main className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">Calculator</h1>
          <p className="text-neutral">Perform calculations with ease</p>
        </div>
        <Calculator />
      </div>
    </main>
  )
}
