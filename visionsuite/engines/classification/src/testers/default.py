    # if args.test_only:
    #     # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #     if model_ema:
    #         evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
    #     else:
    #         evaluate(model, criterion, data_loader_test, device=device)
    #     return