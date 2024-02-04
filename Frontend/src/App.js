import React from 'react'
import Home from './pages/home'
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'
import 'antd/dist/antd.css'
import './App.scss'
import Container from './components/Container'

function App() {
  return (
    <div className="app">
      <Router>
        <Routes>
          <Route exact path="/" element={<Home />}></Route>
          <Route exact path="/paint" element={<Container />}></Route>
        </Routes>
      </Router>
    </div>
  )
}

export default App
