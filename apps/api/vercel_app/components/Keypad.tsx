'use client'

interface KeypadProps {
  onNumberClick: (num: string) => void
  onDecimal: () => void
  onOperation: (op: string) => void
  onEquals: () => void
  onClear: () => void
  onSquareRoot: () => void
  onToggleSign: () => void
  onBackspace: () => void
}

export default function Keypad({
  onNumberClick,
  onDecimal,
  onOperation,
  onEquals,
  onClear,
  onSquareRoot,
  onToggleSign,
  onBackspace,
}: KeypadProps) {
  return (
    <div className="p-4 bg-slate-800">
      <div className="grid grid-cols-4 gap-2">
        {/* Row 1: Actions */}
        <button
          onClick={onClear}
          className="calculator-button-clear"
          aria-label="Clear"
        >
          C
        </button>
        <button
          onClick={onToggleSign}
          className="calculator-button-action"
          aria-label="Toggle sign"
        >
          ±
        </button>
        <button
          onClick={onSquareRoot}
          className="calculator-button-action"
          aria-label="Square root"
        >
          √
        </button>
        <button
          onClick={() => onOperation('÷')}
          className="calculator-button-operator"
          aria-label="Divide"
        >
          ÷
        </button>

        {/* Row 2: 7, 8, 9, × */}
        <button
          onClick={() => onNumberClick('7')}
          className="calculator-button-number"
          aria-label="Seven"
        >
          7
        </button>
        <button
          onClick={() => onNumberClick('8')}
          className="calculator-button-number"
          aria-label="Eight"
        >
          8
        </button>
        <button
          onClick={() => onNumberClick('9')}
          className="calculator-button-number"
          aria-label="Nine"
        >
          9
        </button>
        <button
          onClick={() => onOperation('×')}
          className="calculator-button-operator"
          aria-label="Multiply"
        >
          ×
        </button>

        {/* Row 3: 4, 5, 6, - */}
        <button
          onClick={() => onNumberClick('4')}
          className="calculator-button-number"
          aria-label="Four"
        >
          4
        </button>
        <button
          onClick={() => onNumberClick('5')}
          className="calculator-button-number"
          aria-label="Five"
        >
          5
        </button>
        <button
          onClick={() => onNumberClick('6')}
          className="calculator-button-number"
          aria-label="Six"
        >
          6
        </button>
        <button
          onClick={() => onOperation('-')}
          className="calculator-button-operator"
          aria-label="Subtract"
        >
          −
        </button>

        {/* Row 4: 1, 2, 3, + */}
        <button
          onClick={() => onNumberClick('1')}
          className="calculator-button-number"
          aria-label="One"
        >
          1
        </button>
        <button
          onClick={() => onNumberClick('2')}
          className="calculator-button-number"
          aria-label="Two"
        >
          2
        </button>
        <button
          onClick={() => onNumberClick('3')}
          className="calculator-button-number"
          aria-label="Three"
        >
          3
        </button>
        <button
          onClick={() => onOperation('+')}
          className="calculator-button-operator"
          aria-label="Add"
        >
          +
        </button>

        {/* Row 5: 0, ., Backspace, = */}
        <button
          onClick={() => onNumberClick('0')}
          className="calculator-button-number col-span-2"
          aria-label="Zero"
        >
          0
        </button>
        <button
          onClick={onDecimal}
          className="calculator-button-number"
          aria-label="Decimal point"
        >
          .
        </button>
        <button
          onClick={onBackspace}
          className="calculator-button-action"
          aria-label="Backspace"
        >
          ⌫
        </button>

        {/* Row 6: Advanced operations */}
        <button
          onClick={() => onOperation('%')}
          className="calculator-button-operator"
          aria-label="Modulo"
        >
          %
        </button>
        <button
          onClick={() => onOperation('^')}
          className="calculator-button-operator"
          aria-label="Power"
        >
          ^
        </button>
        <button
          onClick={onEquals}
          className="calculator-button-equals"
          aria-label="Equals"
        >
          =
        </button>
      </div>
    </div>
  )
}
