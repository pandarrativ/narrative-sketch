import React, { Component } from 'react'
import { Button } from 'antd'
import './index.scss'
import { Link } from 'react-router-dom'
import axios from 'axios'

class Home extends Component {
  async orderClick() {
    //method1执行完成后执行method2
    await this.method1()
    var timeID = window.setTimeout(this.method2, 300) 
  }
  method1() {
    console.log(1)
    const home_data = {
      userName: 'bopan'
    }
    axios.post('/api/init', home_data).catch((err) => console.log(err))
    console.log(2)
  }
  method2() {
    console.log(3)
    axios
      .post('/api/newState', { stateType: 'Preparation', prevStateName: 'Dummy' })
      .catch((err) => console.log(err))
      console.log(4)
  }

  // startState() {
  //   axios
  //     .post('/api/newState', { stateType: 'Preparation', prevStateName: 'Dummy' })
  //     .catch((err) => console.log(err))
  // }

  render() {
    return (
      <div className="page">
        {/* <img src={bg} className="background"></img> */}
        <div className="top_info">
          <div>NaSketch</div>
        </div>
        <div className="background_img">
          <div className="description">Drawing to Augment Creativity and Expression</div>
          <Link className="link" to="/paint">
            <Button
              type="primary"
              onClick={() => {
                this.orderClick() //有bug：不能控制好先后触发顺序
                // this.startState()
              }}
            >
              Start Creating
            </Button>
          </Link>
        </div>
      </div>
    )
  }
}

export default Home
