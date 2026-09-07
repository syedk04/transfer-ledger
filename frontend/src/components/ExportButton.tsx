// window.print() + a print stylesheet (see index.css's @media print block)
// rather than html2canvas/jspdf - chosen as the MVP: free, zero extra
// dependency weight, and "print to PDF" is a browser-native affordance
// users already know. A client-side PDF-rendering library is a reasonable
// stretch goal if chart/font fidelity in the printed output ever becomes a
// real complaint, but isn't worth the bundle weight up front.
export function ExportButton() {
  return (
    <button
      type="button"
      onClick={() => window.print()}
      className="no-print rounded border border-gray-300 px-3 py-1.5 text-sm font-medium hover:bg-gray-100 dark:border-gray-600 dark:hover:bg-gray-800"
    >
      Export (print / PDF)
    </button>
  )
}
