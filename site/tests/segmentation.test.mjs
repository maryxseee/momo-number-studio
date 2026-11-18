import test from 'node:test';
import assert from 'node:assert/strict';
import {segmentNumber,validateNumber} from '../engine.mjs';

test('strict range preserves invalid predictions',()=>{
  for(const value of ['1','407','999','1000'])assert.equal(validateNumber(value).valid,true);
  for(const value of ['0','01','1001','9999','', '1e2'])assert.equal(validateNumber(value).valid,false);
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
test('empty canvas and too many glyphs are rejected',()=>{
  assert.throws(()=>segmentNumber(new Uint8Array(840*240),840,240));
  const p=new Uint8Array(840*240);
  for(let n=0;n<5;n++)for(let y=60;y<190;y++)for(let x=30+120*n;x<50+120*n;x++)p[y*840+x]=255;
  assert.throws(()=>segmentNumber(p,840,240),/чотирьох/);
});
