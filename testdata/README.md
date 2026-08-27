# testdata

## `hand_640x360.nv12`

One raw NV12 frame straight off this machine's camera: 640×360, stride 640,
Y plane then interleaved UV, 345 600 bytes. BT.709, limited range. Captured
with

```sh
build/hello_hand --crop --stage landmark --dump-frame testdata/hand_640x360.nv12
```

which writes the frame only when a detection actually happened, so the
fixture is a real camera frame rather than something staged.

### What it is for, and what it is not

It exists so the stage-five tests are **reproducible**. A live camera shows
something different every run, which is fine for a demo and useless for a
test: `hand_cpu`, `hand_frame_agree` and `hand_all` all need the same bytes
every time to mean anything.

**It does not contain a hand the pipeline can reach**, and that is worth
stating plainly rather than discovering later. The palm detector fires on
this frame at score 0.67 — on a *face*. The hand that is in the frame sits
far enough right that `--crop` (which takes the centre 360×360 of a 16:9
frame) cuts it off, and `--letterbox` leaves it too small to clear the
threshold. Dump what the landmark model receives and you see an eye and an
ear:

```sh
build/hello_hand --frame testdata/hand_640x360.nv12 --size 640x360 --crop \
                 --stage landmark --ppm /tmp/crop.ppm
#   presence 0.243   <- the landmark model, correctly, says "not a hand"
```

So the frame is a perfectly good deterministic input for the things the
tests actually check —

- the WebGPU FrameToTensor pass agrees with its C twin to ~1e-5,
- the two execution providers agree with each other to ~1e-3 of a pixel,
- the whole pipeline runs without a stale or absent output —

and it is **not** evidence that detection works. That evidence is a window
with a hand in front of it, and `presence` above 0.5.

`--require-hand` makes "both models agree there is a hand" part of the exit
code. It is deliberately off for this fixture. If you capture one with a
hand near the centre of the frame, turn it on for that fixture's tests: it
is the stronger assertion and worth having.

### The lesson that earned this file

The first read of this fixture reported 21 landmarks in a plausible hand
shape — fingertips above the wrist, each finger chain monotonic — and the
low `presence` beside them got explained away as "the hand must be clipped
at the frame edge". It was not clipped; it was a face, and the landmark
model was doing what it does with a crop that has no hand in it: returning
a hand-shaped answer anyway, with the score that says not to trust it.

`presence` is not a diagnostic to interpret. It is the answer.
