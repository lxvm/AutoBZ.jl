module WannierIOExt

using WannierIO
using AutoBZ

AutoBZ.read_w90_hrdat(io) = WannierIO.read_w90_hrdat(io)
AutoBZ.read_w90_rdat(io) = WannierIO.read_w90_rdat(io)
AutoBZ.read_wout(io) = WannierIO.read_wout(io)

end
