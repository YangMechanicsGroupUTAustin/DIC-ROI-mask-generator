function imageSmoothingGUI
    % Create and show the GUI
    fig = uifigure('Name', 'Image Processing GUI', 'Position', [100 100 1000 600]);
    
    % Store figure handle in UserData to prevent garbage collection
    fig.UserData.figHandle = fig;
    
    % Set close request function to properly handle GUI closure
    fig.CloseRequestFcn = @(src,event) closeFigure(src);
    
    % Create tab group
    tg = uitabgroup(fig, 'Position', [0 0 1000 600]);
    
    % Spatial Smoothing Tab
    tabSpatial = uitab(tg, 'Title', 'Spatial Smoothing');
    createSpatialSmoothingTab(tabSpatial, fig);
    
    % Temporal Smoothing Tab
    tabTemporal = uitab(tg, 'Title', 'Temporal Smoothing');
    createTemporalSmoothingTab(tabTemporal, fig);
    
    % Keep the figure reference alive
    assignin('base', 'imageProcessingGUI_Handle', fig);
end

function closeFigure(fig)
    % Clean up when closing the figure
    try
        evalin('base', 'clear imageProcessingGUI_Handle');
    catch
        % Variable might not exist, ignore error
    end
    delete(fig);
end

function createSpatialSmoothingTab(tab, fig)
    % Input Directory
    uilabel(tab, 'Position', [20 520 100 22], 'Text', 'Input Directory:');
    inputDirEdit = uieditfield(tab, 'Position', [120 520 700 22]);
    uibutton(tab, 'Position', [830 520 150 22], 'Text', 'Browse', ...
        'ButtonPushedFcn', @(btn,event) browseInputDir(inputDirEdit));
    % Output Directory
    uilabel(tab, 'Position', [20 490 100 22], 'Text', 'Output Directory:');
    outputDirEdit = uieditfield(tab, 'Position', [120 490 700 22]);
    uibutton(tab, 'Position', [830 490 150 22], 'Text', 'Browse', ...
        'ButtonPushedFcn', @(btn,event) browseOutputDir(outputDirEdit));
    % Overwrite Checkbox
    overwriteCheck = uicheckbox(tab, 'Position', [120 460 200 22], ...
        'Text', 'Overwrite in Input Directory', ...
        'ValueChangedFcn', @(cb,event) toggleOverwrite(cb, outputDirEdit, inputDirEdit));
    % Parameters
    uilabel(tab, 'Position', [20 430 100 22], 'Text', 'Iterations:');
    iterationsEdit = uieditfield(tab, 'numeric', 'Position', [120 430 100 22], 'Value', 50);
    uilabel(tab, 'Position', [230 430 100 22], 'Text', 'Time Step (dt):');
    dtEdit = uieditfield(tab, 'numeric', 'Position', [330 430 100 22], 'Value', 0.1);
    uilabel(tab, 'Position', [440 430 100 22], 'Text', 'Lambda:');
    lambdaEdit = uieditfield(tab, 'numeric', 'Position', [540 430 100 22], 'Value', 0.5);

    % --- MODIFICATION START: Hiding Viscousness and Surface Tension ---
    % uilabel(tab, 'Position', [20 400 100 22], 'Text', 'Viscousness:');
    % viscousnessEdit = uieditfield(tab, 'numeric', 'Position', [120 400 100 22], 'Value', 0.0);
    % uilabel(tab, 'Position', [230 400 100 22], 'Text', 'Surface Tension:');
    % surfaceTensionEdit = uieditfield(tab, 'numeric', 'Position', [330 400 100 22], 'Value', 0.5);
    % --- MODIFICATION END ---
    
    uilabel(tab, 'Position', [20 400 100 22], 'Text', 'Gaussian Sigma:'); % Adjusted position
    gaussianSigmaEdit = uieditfield(tab, 'numeric', 'Position', [120 400 100 22], 'Value', 4.0); % Adjusted position
    uilabel(tab, 'Position', [230 400 100 22], 'Text', 'Gaussian Size:'); % Adjusted position
    gaussianSizeEdit = uieditfield(tab, 'numeric', 'Position', [330 400 100 22], 'Value', 5); % Adjusted position

    % Real-time Visualization Checkbox
    realTimeVisCheck = uicheckbox(tab, 'Position', [20 370 200 22], ...
        'Text', 'Real-time Visualization', ...
        'ValueChangedFcn', @(cb,event) warnSlowdown(cb));
    
    % Start Button
    uibutton(tab, 'Position', [20 330 960 30], 'Text', 'Start Spatial Smoothing', ...
        'ButtonPushedFcn', @(btn,event) startSpatialSmoothing(fig, tab, inputDirEdit, outputDirEdit, ...
        overwriteCheck, iterationsEdit, dtEdit, lambdaEdit, ...
        gaussianSigmaEdit, gaussianSizeEdit, realTimeVisCheck)); % MODIFIED: Removed viscousnessEdit and surfaceTensionEdit
end

function createTemporalSmoothingTab(tab, fig)
    % Input Directory
    uilabel(tab, 'Position', [20 520 100 22], 'Text', 'Input Directory:');
    inputDirEdit = uieditfield(tab, 'Position', [120 520 700 22]);
    uibutton(tab, 'Position', [830 520 150 22], 'Text', 'Browse', ...
        'ButtonPushedFcn', @(btn,event) browseInputDir(inputDirEdit));
    % Output Directory
    uilabel(tab, 'Position', [20 490 100 22], 'Text', 'Output Directory:');
    outputDirEdit = uieditfield(tab, 'Position', [120 490 700 22]);
    uibutton(tab, 'Position', [830 490 150 22], 'Text', 'Browse', ...
        'ButtonPushedFcn', @(btn,event) browseOutputDir(outputDirEdit));
    % Overwrite Checkbox
    overwriteCheck = uicheckbox(tab, 'Position', [120 460 200 22], ...
        'Text', 'Overwrite in Input Directory', ...
        'ValueChangedFcn', @(cb,event) toggleOverwrite(cb, outputDirEdit, inputDirEdit));
    % Parameters
    uilabel(tab, 'Position', [20 430 100 22], 'Text', 'Variance Threshold:');
    varianceThresholdEdit = uieditfield(tab, 'numeric', 'Position', [120 430 100 22], 'Value', 50000);
    uilabel(tab, 'Position', [230 430 120 22], 'Text', 'Number of Neighbors:');
    numNeighborsEdit = uieditfield(tab, 'numeric', 'Position', [350 430 100 22], 'Value', 2);
    uilabel(tab, 'Position', [460 430 100 22], 'Text', 'Gaussian Sigma:');
    sigmaEdit = uieditfield(tab, 'numeric', 'Position', [560 430 100 22], 'Value', 2);
    uilabel(tab, 'Position', [670 430 100 22], 'Text', '3D Kernel Size:');
    kernelSizeEdit = uieditfield(tab, 'numeric', 'Position', [770 430 100 22], 'Value', 5);
    % Start Button
    uibutton(tab, 'Position', [20 390 960 30], 'Text', 'Start Temporal Smoothing', ...
        'ButtonPushedFcn', @(btn,event) startTemporalSmoothing(fig, tab, inputDirEdit, outputDirEdit, ...
        overwriteCheck, varianceThresholdEdit, numNeighborsEdit, sigmaEdit, kernelSizeEdit));
end

% Helper Functions
function browseInputDir(edit)
    try
        folder = uigetdir('Select Input Directory');
        if folder ~= 0
            edit.Value = folder;
        end
    catch ME
        % Handle any errors that might occur during folder selection
        warning('Error selecting input directory: %s', ME.message);
    end
end

function browseOutputDir(edit)
    try
        folder = uigetdir('Select Output Directory');
        if folder ~= 0
            edit.Value = folder;
        end
    catch ME
        % Handle any errors that might occur during folder selection
        warning('Error selecting output directory: %s', ME.message);
    end
end

function toggleOverwrite(cb, outputEdit, inputDirEdit)
    if cb.Value
        outputEdit.Value = inputDirEdit.Value;
        outputEdit.Enable = 'off';
    else
        outputEdit.Enable = 'on';
    end
end

function warnSlowdown(cb)
    if cb.Value
        fig = ancestor(cb, 'figure');
        if isempty(fig)
            warndlg('Real-time visualization may significantly slow down the process.', 'Performance Warning');
        else
            uialert(fig, 'Real-time visualization may significantly slow down the process.', 'Performance Warning', 'Icon', 'warning');
        end
    end
end

% MODIFIED: Function definition updated to remove viscousnessEdit and surfaceTensionEdit
function startSpatialSmoothing(fig, tab, inputDirEdit, outputDirEdit, overwriteCheck, ...
        iterationsEdit, dtEdit, lambdaEdit, ...
        gaussianSigmaEdit, gaussianSizeEdit, realTimeVisCheck)
    % Get parameters
    inputDir = inputDirEdit.Value;
    if overwriteCheck.Value
        outputDir = inputDir;
    else
        outputDir = outputDirEdit.Value;
    end
    params.numIterations = iterationsEdit.Value;
    params.dt = dtEdit.Value;
    params.lambda = lambdaEdit.Value;
    % --- MODIFICATION: Hardcode viscousness and surface tension to 0 ---
    params.viscousness = 0.0;
    params.surfaceTension = 0.0;
    % --- END MODIFICATION ---
    params.gaussianSigma = gaussianSigmaEdit.Value;
    params.gaussianSize = gaussianSizeEdit.Value;
    params.realTimeVis = realTimeVisCheck.Value;
    
    % Perform smoothing
    [success, message] = smoothImages(inputDir, outputDir, params, fig);
    
    % Only proceed if smoothing was successful
    if success
        % Ask to view results and generate video
        choice = uiconfirm(fig, 'Spatial smoothing completed. Do you want to view results?', ...
            'View Results', 'Options', {'Yes', 'No'});
        if strcmp(choice, 'Yes')
            viewResults(inputDir, outputDir);
        end
        choice = uiconfirm(fig, 'Do you want to generate a comparison video?', ...
            'Generate Video', 'Options', {'Yes', 'No'});
        if strcmp(choice, 'Yes')
            generateVideo(inputDir, outputDir);
        end
    else
        % Display message if smoothing was not successful
        uialert(fig, message, 'Smoothing Incomplete', 'Icon', 'warning');
    end
end

function startTemporalSmoothing(fig, tab, inputDirEdit, outputDirEdit, overwriteCheck, ...
        varianceThresholdEdit, numNeighborsEdit, sigmaEdit, kernelSizeEdit)
    inputDir = inputDirEdit.Value;
    if overwriteCheck.Value
        outputDir = inputDir;
    else
        outputDir = outputDirEdit.Value;
    end
    varianceThreshold = varianceThresholdEdit.Value;
    numNeighbors = numNeighborsEdit.Value;
    sigma = sigmaEdit.Value;
    kernelSize = kernelSizeEdit.Value;
    
    temporalSmoothingProcess(inputDir, outputDir, varianceThreshold, numNeighbors, sigma, kernelSize, fig);
end

function [success, message] = smoothImages(inputDir, outputDir, params, fig)
    % Get image files
    imageFiles = getImageFiles(inputDir);
    if isempty(imageFiles)
        success = false;
        message = 'No supported image files found in the input directory.';
        uialert(fig, message, 'Error', 'Icon', 'error');
        return;
    end
    
    % Create progress bar
    d = uiprogressdlg(fig, 'Title', 'Processing Images', ...
        'Message', 'Starting...', ...
        'Cancelable', 'on');
        
    % Process each image
    for imgInd = 1:numel(imageFiles)
        if d.CancelRequested
            success = false;
            message = 'Operation cancelled by user.';
            close(d);
            return;
        end
        % Update progress
        d.Value = imgInd / numel(imageFiles);
        d.Message = sprintf('Processing image %d of %d', imgInd, numel(imageFiles));
        
        % Read and process image
        imagePath = fullfile(inputDir, imageFiles(imgInd).name);
        [originalImage, ~] = imread(imagePath);
        originalImage = im2double(originalImage);
        
        % Smooth image
        smoothedImage = smoothSingleImage(originalImage, params);
        
        % Save smoothed image
        outputFilePath = fullfile(outputDir, imageFiles(imgInd).name);
        imwrite(smoothedImage, outputFilePath); 
    end
    
    % Close progress bar
    close(d);
    success = true;
    message = 'Smoothing completed successfully.';
end

function smoothedImage = smoothSingleImage(originalImage, params)
    % Initialize smoothed image
    smoothedImage = originalImage;
    
    % Create a Laplacian kernel for divergence calculation
    laplacianKernel = [0 1 0; 1 -4 1; 0 1 0];
    
    % Apply Perona-Malik equation for image smoothing
    for iteration = 1:params.numIterations
        % Calculate gradient and diffusivity
        [Gx, Gy] = gradient(smoothedImage);
        diffusivity = exp(-(Gx.^2 + Gy.^2) / (2 * params.lambda^2));
        
        % Calculate Laplacian of diffusivity
        laplacianDiffusivity = imfilter(diffusivity, laplacianKernel, 'symmetric');
        
        % Update smoothed image
        smoothedImage = smoothedImage + params.dt * (laplacianDiffusivity .* del2(smoothedImage));
            
        % Apply Gaussian filter
        smoothedImage = imgaussfilt(smoothedImage, params.gaussianSigma, 'FilterSize', params.gaussianSize);
        
        % Real-time visualization (if selected)
        if params.realTimeVis
            displayResults(originalImage, smoothedImage, smoothedImage > 0.5, iteration);
            drawnow;
        end
    end
    
    % Create final binary mask
    smoothedImage = smoothedImage(:,:,1) > 0.5;
    % Instead of creating a binary mask, scale the smoothed image to match the original format
    smoothedImage = im2uint8(smoothedImage);  % For 8-bit images
    % If you need to handle other bit depths:
    % smoothedImage = im2uint16(smoothedImage);  % For 16-bit images
    % smoothedImage = im2double(smoothedImage);  % For double precision floating point
end

function imageFiles = getImageFiles(directory)
    % List of supported image formats
    formats = {'*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.gif'};
    imageFiles = [];
    for i = 1:length(formats)
        imageFiles = [imageFiles; dir(fullfile(directory, formats{i}))];
    end
    
    % Sort files by name
    [~, index] = sort({imageFiles.name});
    imageFiles = imageFiles(index);
end

function displayResults(originalImage, smoothedImage, smoothedImageMask, iteration)
    figure(2);
    subplot(1, 2, 1);
    imshow(originalImage);
    title('Original Image');
    
    subplot(1, 2, 2);
    imshow(smoothedImageMask);
    title(['Smoothed Mask at Iter ', num2str(iteration)]);
    
    sgtitle(['Processing Image: Iteration ', num2str(iteration)]);
end

function viewResults(inputDir, outputDir)
    % Get image files
    inputFiles = getImageFiles(inputDir);
    outputFiles = getImageFiles(outputDir);
    
    % Create figure for viewing results
    fig = figure('Name', 'Smoothing Results', 'NumberTitle', 'off', 'Position', [100 100 800 600]);
    
    % Create UI controls
    uicontrol('Style', 'pushbutton', 'String', 'Next', ...
        'Position', [20 20 50 20], 'Callback', @nextImage);
    uicontrol('Style', 'pushbutton', 'String', 'Previous', ...
        'Position', [80 20 50 20], 'Callback', @prevImage);
        
    % Add image selection input
    uicontrol('Style', 'text', 'String', 'Go to Image:', ...
        'Position', [140 20 70 20]);
    imageSelectEdit = uicontrol('Style', 'edit', ...
        'Position', [220 20 50 20]);
    uicontrol('Style', 'pushbutton', 'String', 'Go', ...
        'Position', [280 20 30 20], 'Callback', @goToImage);
        
    % Add slider for quick navigation
    slider = uicontrol('Style', 'slider', ...
        'Min', 1, 'Max', numel(inputFiles), 'Value', 1, ...
        'SliderStep', [1/(numel(inputFiles)-1), 10/(numel(inputFiles)-1)], ...
        'Position', [20 50 760 20], ...
        'Callback', @sliderCallback);
        
    % Add ButtonDownFcn and ButtonUpFcn to the slider
    set(slider, 'ButtonDownFcn', @startDragging);
    set(fig, 'WindowButtonUpFcn', @stopDragging);
    set(fig, 'WindowButtonMotionFcn', @dragging);
    
    % Add text to show current image number
    textHandle = uicontrol('Style', 'text', ...
        'Position', [350 20 100 20], ...
        'String', '1 / ' + string(numel(inputFiles)));
        
    % Initialize image index and dragging flag
    currentIndex = 1;
    isDragging = false;
    
    % Display first image
    displayImage(currentIndex);
    
    function nextImage(~, ~)
        currentIndex = min(currentIndex + 1, numel(inputFiles));
        updateDisplay();
    end
    
    function prevImage(~, ~)
        currentIndex = max(currentIndex - 1, 1);
        updateDisplay();
    end
    
    function goToImage(~, ~)
        selectedIndex = str2double(imageSelectEdit.String);
        if ~isnan(selectedIndex) && selectedIndex >= 1 && selectedIndex <= numel(inputFiles)
            currentIndex = selectedIndex;
            updateDisplay();
        else
            errordlg('Please enter a valid image number.', 'Invalid Input');
        end
    end
    
    function sliderCallback(hObject, ~)
        currentIndex = round(get(hObject, 'Value'));
        updateDisplay();
    end
    
    function startDragging(~, ~)
        isDragging = true;
    end
    
    function stopDragging(~, ~)
        isDragging = false;
    end
    
    function dragging(~, ~)
        if isDragging
            currentIndex = round(get(slider, 'Value'));
            updateDisplay();
        end
    end
    
    function updateDisplay()
        displayImage(currentIndex);
        set(slider, 'Value', currentIndex);
        set(textHandle, 'String', sprintf('%d / %d', currentIndex, numel(inputFiles)));
    end
    
    function displayImage(index)
        originalImage = imread(fullfile(inputDir, inputFiles(index).name));
        smoothedImage = imread(fullfile(outputDir, outputFiles(index).name));
        subplot(1, 2, 1);
        imshow(originalImage);
        title('Original Image');
        subplot(1, 2, 2);
        imshow(smoothedImage);
        title('Smoothed Image');
        sgtitle(['Image ' num2str(index) '/' num2str(numel(inputFiles))]);
    end
end

function generateVideo(inputDir, outputDir)
    % Get image files
    inputFiles = getImageFiles(inputDir);
    outputFiles = getImageFiles(outputDir);
    
    % Prompt user for video settings
    prompt = {'Enter frame rate (fps):', 'Enter video format (mp4, avi, mov):'};
    dlgtitle = 'Video Settings';
    dims = [1 35];
    definput = {'5', 'mp4'};
    answer = inputdlg(prompt, dlgtitle, dims, definput);
    
    if isempty(answer)
        return; % User cancelled
    end
    
    fps = str2double(answer{1});
    format = answer{2};
    
    % Let user choose save location and filename
    [filename, pathname] = uiputfile(['*.' format], 'Save Video As');
    if isequal(filename, 0) || isequal(pathname, 0)
        return; % User cancelled
    end
    videoFile = fullfile(pathname, filename);
    
    % Create video writer object
    if strcmpi(format, 'mp4')
        v = VideoWriter(videoFile, 'MPEG-4');
    else
        v = VideoWriter(videoFile);
    end
    v.FrameRate = fps;
    open(v);
    
    % Create progress bar
    progressBar = waitbar(0, 'Generating Video...');
    
    % Create figure for video frames
    fig = figure('Visible', 'on', 'Position', [100, 100, 1000, 500]);  % Make figure visible and larger
    
    % Generate video frames
    for i = 1:numel(inputFiles)
        originalImage = imread(fullfile(inputDir, inputFiles(i).name));
        smoothedImage = imread(fullfile(outputDir, outputFiles(i).name));
        
        subplot(1, 2, 1);
        imshow(originalImage);
        title('Original Image');
        
        subplot(1, 2, 2);
        imshow(smoothedImage);
        title('Smoothed Image');
        
        sgtitle(['Frame ' num2str(i) '/' num2str(numel(inputFiles))]);
        
        % Ensure the figure is rendered
        drawnow;
        
        % Capture the plot as an image 
        frame = getframe(fig); 
        
        % Write the image to video
        writeVideo(v, frame);
        
        % Update progress bar
        waitbar(i / numel(inputFiles), progressBar);
    end
    
    % Close the video writer object
    close(v);
    
    % Close the figure and progress bar
    close(fig);
    close(progressBar);
    
    msgbox(['Video saved as ' videoFile], 'Video Generation Complete');
end

function temporalSmoothingProcess(inputDir, outputDir, varianceThreshold, numNeighbors, sigma, kernelSize, fig)
    try
        % Load images
        imageFiles = getImageFiles2(inputDir);
        numFrames = numel(imageFiles);
        
        if numFrames == 0
            errordlg('No supported image files found in the input directory.', 'Error');
            return;
        end
        
        % Initialize progress bar
        d = uiprogressdlg(fig, 'Title', 'Processing Images', 'Message', 'Starting...', 'Cancelable', 'on');
        
        % Load and preprocess images
        frames = loadAndPreprocessImages(imageFiles, inputDir, d);
        if isempty(frames), return; end
        
        % Identify and remove bad frames
        [imageSequence, badFrameIndices] = removeBadFrames(frames, varianceThreshold);
        
        % Fill NaN values and apply temporal smoothing
        filledImageSequence = fillNaNValues(imageSequence, numNeighbors, sigma, d);
        
        % Apply 3D Gaussian filter
        smoothedImageSequence = apply3DGaussianFilter(filledImageSequence, kernelSize, sigma, d);
        
        % Threshold and save filtered frames
        saveFilteredFrames(smoothedImageSequence, outputDir, imageFiles, d);
        
        % Close progress dialog
        close(d);
        
        msgbox('Temporal smoothing completed successfully!', 'Success');
    catch e
        errordlg(['Error: ' e.message], 'Error');
    end
end

function frames = loadAndPreprocessImages(imageFiles, inputDir, d)
    numFrames = numel(imageFiles);
    frames = cell(1, numFrames);
    for i = 1:numFrames
        if d.CancelRequested, frames = {}; return; end
        d.Value = i / numFrames; d.Message = sprintf('Loading image %d of %d', i, numFrames);
        currentFrame = imread(fullfile(inputDir, imageFiles(i).name));
        if size(currentFrame, 3) > 1
            currentFrame = rgb2gray(currentFrame);
        end
        frames{i} = double(currentFrame);
    end
end

function imageFiles = getImageFiles2(directory)
    % List of supported image formats
    formats = {'*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.gif'};
    
    imageFiles = [];
    for i = 1:length(formats)
        imageFiles = [imageFiles; dir(fullfile(directory, formats{i}))];
    end
    
    % Sort files by name
    [~, index] = sort({imageFiles.name});
    imageFiles = imageFiles(index);
end

function [imageSequence, badFrameIndices] = removeBadFrames(frames, varianceThreshold)
    numFrames = numel(frames);
    imageSize = size(frames{1});
    imageSequence = zeros([imageSize, numFrames]);
    variances = zeros(1, numFrames);
    
    for i = 1:numFrames
        imageSequence(:, :, i) = frames{i};
        variances(i) = var(frames{i}(:));
    end
    
    badFrameIndices1 = find(variances > varianceThreshold);
    badFrameIndices2 = find(variances == 0);
    neighborIndices = [badFrameIndices2-1, badFrameIndices2+1];
    neighborIndices = neighborIndices(neighborIndices >= 1 & neighborIndices <= numFrames);
    badFrameIndices = unique([badFrameIndices1, badFrameIndices2, neighborIndices]);
    imageSequence(:,:,badFrameIndices) = nan;
end

function filledImageSequence = fillNaNValues(imageSequence, numNeighbors, sigma, d)
    filledImageSequence = imageSequence;
    numFrames = size(imageSequence, 3);
    
    for frameIndex = 1:numFrames
        if d.CancelRequested, return; end
        d.Value = frameIndex / numFrames; d.Message = sprintf('Filling NaN values %d of %d', frameIndex, numFrames);
        
        frame = imageSequence(:, :, frameIndex);
        nanMask = isnan(frame);
        
        if any(nanMask(:))
            neighborIndices = max(1, frameIndex - numNeighbors):min(numFrames, frameIndex + numNeighbors);
            neighborFrames = imageSequence(:, :, neighborIndices);
            
            for tempi = 1:size(neighborFrames, 3)
                tempframe = neighborFrames(:,:,tempi);
                tempframe = imgaussfilt(tempframe, sigma);
                neighborFrames(:,:,tempi) = tempframe;
            end
            
            temporalAverage = nanmean(neighborFrames, 3);
            filledImageSequence(:, :, frameIndex) = frame;
            filledImageSequence(:, :, frameIndex) =  reshape(temporalAverage(nanMask),  size(filledImageSequence,1) , size(filledImageSequence,2), 1);
        end
    end
end

function smoothedImageSequence = apply3DGaussianFilter(filledImageSequence, kernelSize, sigma, d)
    gaussianKernel = fspecial3('gaussian', [kernelSize, kernelSize, kernelSize], sigma);
    smoothedImageSequence = filledImageSequence;
    numFrames = size(filledImageSequence, 3);
    
    for frameIndex = 1:numFrames
        if d.CancelRequested, return; end
        d.Value = frameIndex / numFrames; d.Message = sprintf('Applying 3D Gaussian filter %d of %d', frameIndex, numFrames);
        
        frame = filledImageSequence(:, :, frameIndex);
        smoothedFrame = imfilter(frame, gaussianKernel, 'conv', 'replicate');
        smoothedImageSequence(:, :, frameIndex) = smoothedFrame;
    end
end

function saveFilteredFrames(smoothedImageSequence, outputDir, originalFiles, d)
    numFrames = size(smoothedImageSequence, 3);
    mkdir(outputDir);
    
    for frameIndex = 1:numFrames
        if d.CancelRequested, return; end
        d.Value = frameIndex / numFrames; d.Message = sprintf('Saving filtered frame %d of %d', frameIndex, numFrames);
        
        % Read the original image to get its format and info
        originalFramePath = fullfile(originalFiles(frameIndex).folder, originalFiles(frameIndex).name);
        originalFrame = imread(originalFramePath);
        imageInfo = imfinfo(originalFramePath);
        
        % Get the smoothed frame
        smoothedFrame = smoothedImageSequence(:, :, frameIndex);
        
        % Convert smoothed frame to match the original image class
        switch class(originalFrame)
            case 'uint8'
                smoothedFrame = im2uint8(smoothedFrame);
            case 'uint16'
                smoothedFrame = im2uint16(smoothedFrame);
            case 'double'
                % Already in double format, no conversion needed
            otherwise
                warning('Unsupported image format. Converting to uint8.');
                smoothedFrame = im2uint8(smoothedFrame);
        end
        
        % Preserve color channels if present
        if size(originalFrame, 3) > 1
            smoothedFrame = repmat(smoothedFrame, [1, 1, size(originalFrame, 3)]);
        end
        
        [~, name, ext] = fileparts(originalFiles(frameIndex).name);
        outputFilePath = fullfile(outputDir, [name, ext]);
        
        % Write the image, preserving resolution if available
        if isfield(imageInfo, 'XResolution') && isfield(imageInfo, 'YResolution')
            imwrite(smoothedFrame, outputFilePath, 'Resolution', [imageInfo.XResolution imageInfo.YResolution]);
        else
            imwrite(smoothedFrame, outputFilePath);
        end
    end
end