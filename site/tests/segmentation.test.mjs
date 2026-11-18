import test from 'node:test';
import assert from 'node:assert/strict';
import {segmentNumber,validateNumber,rankSequences} from '../engine.mjs';

test('arbitrary digit strings preserve leading zeros and precision',()=>{
  for(const value of ['0','01','407','1001','999999999999999999999999999999']){
    const result=validateNumber(value);assert.equal(result.valid,true);assert.equal(result.number,value);
  }
  for(const value of ['', '1e2','4.7','-3','hello'])assert.equal(validateNumber(value).valid,false);
});
test('segmentation preserves order, joins detached strokes, removes isolated specks',()=>{
  const width=840,height=240,pixels=new Uint8Array(width*height);
  for(let y=60;y<190;y++)for(let x=70;x<90;x++)pixels[y*width+x]=255;
  for(let y=60;y<190;y++)for(let x=210;x<280;x++)if(x<260||x>=264)pixels[y*width+x]=255;
  pixels[2*width+700]=255;
  const parts=segmentNumber(pixels,width,height);
  assert.equal(parts.length,2);
  assert.deepEqual(parts.map(p=>p.bbox),[[70,60,20,130],[210,60,70,130]]);
});
test('empty canvas is rejected but more than four digits are supported',()=>{
  assert.throws(()=>segmentNumber(new Uint8Array(840*240),840,240));
  const p=new Uint8Array(840*240);
  for(let n=0;n<5;n++)for(let y=60;y<190;y++)for(let x=30+120*n;x<50+120*n;x++)p[y*840+x]=255;
  assert.equal(segmentNumber(p,840,240).length,5);
});
test('whole-number alternatives follow joint digit scores',()=>{
  const a=Array(10).fill(0),b=Array(10).fill(0);a[4]=.6;a[5]=.4;b[7]=.7;b[8]=.3;
  const ranked=rankSequences([a,b]);assert.deepEqual(ranked.map(r=>r.text),['47','57','48']);
  assert.equal(ranked[0].relativeScore,1);assert.ok(ranked[1].relativeScore>ranked[2].relativeScore);
});
test('log-space ranking remains finite for long strings and tie order is stable',()=>{
  const p=Array(10).fill(.1),ranked=rankSequences(Array(400).fill(p));
  assert.equal(ranked.length,3);assert.equal(ranked[0].text.length,400);
  assert.ok(ranked.every(r=>Number.isFinite(r.logScore)&&r.relativeScore>0));
  assert.equal(new Set(ranked.map(r=>r.text)).size,3);
});
