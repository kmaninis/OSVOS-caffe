function partition = ucm2part( ucm2, threshold )
    tmp = (bwlabel(ucm2'<=threshold,4))';
    partition = uint32(tmp(2:2:end,2:2:end));
end

