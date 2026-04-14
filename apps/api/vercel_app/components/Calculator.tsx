'use client'

import { useState } from 'react'
import Display from './Display'
import Keypad from './Keypad'

export default function Calculator() {
  const [display, setDisplay] = useState('0')
  const [previousValue, setPreviousValue] = useState<number | null>(null)
  const [operation, setOperation] = useState<string | null>(null)
  const [waitingForNewValue, setWaitingForNewValue] = useState(false)
  const [history, setHistory] = useState<string[]>([])

  const handleNumberClick = (num: string) => {
    if (waitingForNewValue) {
      setDisplay(num)
      setWaitingForNewValue(false)
    } else {
      setDisplay(display === '0' ? num : display + num)
    }
  }

  const handleDecimal = () => {
    if (waitingForNewValue) {
      setDisplay('0.')
      setWaitingForNewValue(false)
    } else if (!display.includes('.')) {
      setDisplay(display + '.')
    }
  }

  const handleOperation = (op: string) => {
    const currentValue = parseFloat(display)

    if (previousValue === null) {
      setPreviousValue(currentValue)
    } else if (operation) {
      const result = calculate(previousValue, currentValue, operation)
      setDisplay(String(result))
      setPreviousValue(result)
    }

    setOperation(op)
    setWaitingForNewValue(true)
  }

  const calculate = (prev: number, current: number, op: string): number => {
    switch (op) {
      case '+':
        return prev + current
      case '-':
        return prev - current
      case '×':
        return prev * current
      case '÷':
        return prev / current
      case '%':
        return prev % current
      case '^':
        return Math.pow(prev, current)
      default:
        return current
    }
  }

  const handleEquals = () => {
    if (operation && previousValue !== null) {
      const currentValue = parseFloat(display)
      const result = calculate(previousValue, currentValue, operation)
      const historyEntry = `${previousValue} ${operation} ${currentValue} = ${result}`
      setHistory([...history, historyEntry])
      setDisplay(String(result))
      setPreviousValue(null)
      setOperation(null)
      setWaitingForNewValue(true)
    }
  }

  const handleClear = () => {
    setDisplay('0')
    setPreviousValue(null)
    setOperation(null)
    setWaitingForNewValue(false)
  }

  const handleSquareRoot = () => {
    const current = parseFloat(display)
    const result = Math.sqrt(current)
    setDisplay(String(result))
  }

  const handleToggleSign = () => {
    const current = parseFloat(display)
    setDisplay(String(current * -1))
  }

  const handleBackspace = () => {
    if (display.length === 1) {
      setDisplay('0')
    } else {
      setDisplay(display.slice(0, -1))
    }
  }

  return (
    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl shadow-2xl overflow-hidden">
      <Display value={display} history={history} />
      <Keypad
        onNumberClick={handleNumberClick}
        onDecimal={handleDecimal}
        onOperation={handleOperation}
        onEquals={handleEquals}
        onClear={handleClear}
        onSquareRoot={handleSquareRoot}
        onToggleSign={handleToggleSign}
        onBackspace={handleBackspace}
      />
    </div>
  )
}
