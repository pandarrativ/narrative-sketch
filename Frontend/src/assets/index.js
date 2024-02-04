import pre from './images/knowledge.png'
import evaluation from './images/evaluation.png'
import incubation from './images/incubation.png'
import ins_combination from './images/ins_combination.png'
import ins_exploration from './images/ins_exploration.png'
import ins_transformation from './images/ins_transformation.png'
import timeline from './images/timeline.png'
import finish from './images/finish.svg'

const images = require.context('./', true, /\.png$/)

export {
  pre,
  evaluation,
  incubation,
  ins_combination,
  ins_exploration,
  ins_transformation,
  timeline,
  finish,
  images
}
