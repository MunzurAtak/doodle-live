import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'

describe('App', () => {
  it('renders the title, canvas and controls', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Doodle Live' })).toBeInTheDocument()
    expect(screen.getByLabelText('Drawing canvas')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled()
  })
})
