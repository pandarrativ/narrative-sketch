import { useEffect, useState } from 'react'
import {
  pre,
  evaluation,
  incubation,
  ins_combination,
  ins_exploration,
  ins_transformation
} from '../../assets'
import './index.scss'

const Foot = ({ setMenu, menu }) => {
  const [clickable, setClickable] = useState([])

  useEffect(() => {
    switch (menu) {
      case 0:
        setClickable([])
        break
      case 1:
        setClickable([2, 3, 4])
        break
      case 2:
        setClickable([3, 4])
        break
      case 3:
        setClickable([2, 4])
        break
      case 4:
        setClickable([2, 3])
        break
      case 5:
        setClickable([])
        break
      default:
        setClickable([])
        break
    }
  }, [menu])

  return (
    <div className="bottom">
      <div className={`bottom-box${menu === 0 ? ' bottom-box-selected' : ''}`}>
        <img src={pre} className="photo1" alt="" />
        <div>Preparation</div>
      </div>
      <div className={`bottom-box${menu === 1 ? ' bottom-box-selected' : ''}`}>
        <img src={incubation} className="photo" alt="" />
        <div>Incubation</div>
      </div>
      <div
        className={`bottom-box${menu === 2 ? ' bottom-box-selected' : ''}${
          clickable.includes(2) ? ' bottom-box-clickable' : ''
        }`}
        onClick={() => {
          if (clickable.includes(2)) setMenu(2)
        }}
      >
        <img src={ins_combination} className="photo" alt="" />
        <div>Ins-combination</div>
      </div>
      <div
        className={`bottom-box${menu === 3 ? ' bottom-box-selected' : ''}${
          clickable.includes(3) ? ' bottom-box-clickable' : ''
        }`}
        onClick={() => {
          if (clickable.includes(3)) setMenu(3)
        }}
      >
        <img src={ins_exploration} className="photo" alt="" />
        <div>Ins-exploration</div>
      </div>
      <div
        className={`bottom-box${menu === 4 ? ' bottom-box-selected' : ''}${
          clickable.includes(4) ? ' bottom-box-clickable' : ''
        }`}
        onClick={() => {
          if (clickable.includes(4)) setMenu(4)
        }}
      >
        <img src={ins_transformation} className="photo" alt="" />
        <div>Ins-transformation</div>
      </div>
      <div
        className={`bottom-box${menu === 5 ? ' bottom-box-selected' : ''}${
          clickable.includes(5) ? ' bottom-box-clickable' : ''
        }`}
        onClick={() => {
          if (clickable.includes(5)) setMenu(5)
        }}
      >
        <img src={evaluation} className="photo" alt="" />
        <div>Evaluation</div>
      </div>
    </div>
  )
}

export default Foot
