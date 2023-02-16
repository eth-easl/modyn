Two current issues:

1) We cannot make a gRPC connection in the OnlineDataset to the Selector. For some reason, we cannot connect and if we don't check then we segfault as soon as we make a call.
Next debug step would be https://github.com/grpc/grpc/blob/master/doc/fork_support.md
Add environment variable to docker-compose.yml
I did that however i did not verify that it actually arrived at the python process (i think it might have not been applied because i just restarted the container)
need to print all environment variables in trainer server to test

2) for some reason if I dont train with checkpoints the trainer server still checks out some weird path for checkpoints