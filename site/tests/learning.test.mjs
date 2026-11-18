import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {PersonalHead} from '../personal-head.mjs';
const require=createRequire(import.meta.url);
const tf=require('../vendor/tf.min.js');

test('real gradient-based update lowers loss and changes the trained classifier',async()=>{
  await tf.setBackend('cpu');await tf.ready();
  const features=4,classes=10;
  const initial={featureSize:features,kernel:Array(features*classes).fill(0),bias:Array(classes).fill(0)};
  const replay={train:{features:[[0,1,0,0],[0,0,1,0]],labels:[2,3]},guard:{features:[[0,1,0,0],[0,0,1,0]],labels:[2,3]}};
  const head=new PersonalHead(tf,initial,replay,{epochs:20,learningRate:.02,guardTolerance:1});
  const before=head.weights();
  const result=await head.train([{features:[1,0,0,0],label:7}]);
  assert.equal(result.accepted,true);assert.ok(result.lossAfter<result.lossBefore);
  assert.notDeepEqual(head.weights(),before);
  const scores=head.predict([[1,0,0,0]]);
  assert.equal(Array.from(scores).indexOf(Math.max(...scores)),7);
  const saved=head.weights();
  const restored=new PersonalHead(tf,initial,replay,{guardTolerance:1});restored.restore(saved);
  const again=restored.predict([[1,0,0,0]]);
  scores.forEach((v,i)=>assert.ok(Math.abs(v-again[i])<1e-6));
  head.dispose();restored.dispose();
});

test('invalid correction features/labels are rejected before training',async()=>{
  const initial={featureSize:2,kernel:Array(20).fill(0),bias:Array(10).fill(0)};
  const replay={train:{features:[[1,0]],labels:[0]},guard:{features:[[1,0]],labels:[0]}};
  const head=new PersonalHead(tf,initial,replay);
  await assert.rejects(()=>head.train([{features:[NaN,0],label:2}]));
  await assert.rejects(()=>head.train([{features:[1,0],label:12}]));
  head.dispose();
});

test('conflicting correction that fails the guard restores exact previous weights',async()=>{
  const kernel=Array(20).fill(0);kernel[0]=2;
  const initial={featureSize:2,kernel,bias:Array(10).fill(0)};
  const replay={train:{features:[[1,0]],labels:[0]},guard:{features:[[1,0]],labels:[0]}};
  const head=new PersonalHead(tf,initial,replay,{epochs:30,learningRate:.1,guardTolerance:0});
  const before=head.weights(),scores=head.predict([[1,0]]);
  const result=await head.train([{features:[1,0],label:7}]);
  assert.equal(result.accepted,false);assert.equal(result.guardBefore,1);assert.equal(result.guardAfter,0);
  assert.deepEqual(head.weights(),before);assert.deepEqual(head.predict([[1,0]]),scores);
  head.dispose();
});
