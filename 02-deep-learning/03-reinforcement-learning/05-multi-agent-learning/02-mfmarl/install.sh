# Install on OSX

git clone git@github.com:geek-ai/MAgent.git
cd MAgent

brew install cmake llvm boost@1.55
brew install jsoncpp argp-standalone
brew tap david-icracked/homebrew-websocketpp
brew install --HEAD david-icracked/websocketpp/websocketpp
brew link --force boost@1.55

bash build.sh
export PYTHONPATH=$(pwd)/python:$PYTHONPATH